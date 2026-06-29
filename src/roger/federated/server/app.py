"""app.py — FastAPI wire layer for the aggregation server.

Endpoints the client's transport.py speaks:
  GET  /status?model_id=  -> {mode, k_min, k_target}   (which regime to use; see aggregate.mode())
  POST /round/register   {model_id, pubkey(hex)} -> {round_id, peers:[hex,...]}   (long-polls until the
                         cohort seals; 503 if it stays below k_min so the client skips — an empty peer
                         set would make the client upload UNMASKED, leaking its ΔW).
  POST /contribute       octet-stream safetensors ({"masked": int64} + meta) -> 200. The body is
                         STREAMED straight to a per-upload temp object in the store; never buffered whole.
  POST /contribute_dp    octet-stream safetensors (dense ΔW + meta) -> 200   (bootstrap: a single
                         DP-noised UNMASKED ΔW, folded in RAM into the global — no cohort)
  GET  /global?since=&model_id=  -> 204, or 200 streamed octet-stream dense ΔW + X-Cursor header.

Concurrency. /round/register holds the connection on an asyncio.Condition until its round seals — the
barrier that gives every cohort member the *same* frozen peer set (so the pairwise masks cancel). All
round-state mutation happens in the synchronous Aggregator methods with no await between, so under the
single event-loop thread they are atomic. The two I/O-heavy steps are kept OFF that thread via
run_in_threadpool: streaming a body into the store, and `run_finalize` (per-module aggregation). A
per-model lock serializes finalizes of the same model so their global-version bumps don't race.
"""
import asyncio, ipaddress, json, os, struct, time, uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from roger.federated.server import store
from roger.federated.server.aggregate import Aggregator

_STAGE_BATCH = 8 << 20                 # bytes buffered before an off-thread store write (caps RAM/upload)


def _env_set(name: str) -> set | None:
    v = os.environ.get(name, "").strip()
    return {x for x in (s.strip() for s in v.split(",")) if x} or None


def create_app(aggregator: Aggregator | None = None) -> FastAPI:
    """Build the app around an Aggregator (injectable for tests). Config comes from env so the same
    image deploys anywhere: ROGER_AGG_* knobs + ROGER_SERVER_DATA."""
    agg = aggregator or Aggregator(
        datadir=os.environ.get("ROGER_SERVER_DATA", "./server-data"),
        k_min=int(os.environ.get("ROGER_AGG_KMIN", "3")),
        k_target=int(os.environ.get("ROGER_AGG_KTARGET", "5")),
        eta=float(os.environ.get("ROGER_AGG_ETA", "1.0")),
        eta_boot=float(os.environ["ROGER_AGG_ETA_BOOT"]) if os.environ.get("ROGER_AGG_ETA_BOOT") else None,
        clip_norm=float(os.environ.get("ROGER_AGG_CLIP", "1.0")),
        ip_binding=os.environ.get("ROGER_AGG_IPBIND", "1") != "0",
        allowlist=_env_set("ROGER_AGG_MODELS"),
        busy_threshold=int(os.environ["ROGER_AGG_BUSY_THRESHOLD"]) if os.environ.get("ROGER_AGG_BUSY_THRESHOLD") else None,
        busy_window=float(os.environ.get("ROGER_AGG_BUSY_WINDOW", "180")),
    )
    register_window = float(os.environ.get("ROGER_AGG_W", "20"))   # < client 30s register timeout
    collect_window = float(os.environ.get("ROGER_AGG_U", "20"))    # how long to wait for all uploads
    trusted = [ipaddress.ip_network(c.strip()) for c in
               os.environ.get("ROGER_AGG_TRUSTED_PROXIES", "127.0.0.1/32,::1/128,10.0.0.0/8").split(",") if c.strip()]

    app = FastAPI()
    cond = asyncio.Condition()
    finalize_locks: dict[str, asyncio.Lock] = {}   # model_id -> lock; serializes same-model G writes

    def client_ip(req: Request) -> str:
        peer = req.client.host if req.client else ""
        xff = req.headers.get("x-forwarded-for")
        if xff and _trusted(peer, trusted):
            return xff.split(",")[0].strip()   # leftmost = original client when the hop is trusted
        return peer

    async def _finalize(rnd):
        """Claim the round under the lock, then aggregate-or-cleanup off the event loop. Idempotent: the
        inline (all-in-early) trigger and the U-deadline backstop both call this; only one claim wins."""
        async with cond:
            status = agg.claim_finalize(rnd)
            if status != "done":
                lock = finalize_locks.setdefault(rnd.model_id, asyncio.Lock())
        if status == "done":
            return
        async with lock:                       # serialize same-model global writes (version bumps)
            await run_in_threadpool(agg.run_finalize, rnd, status)
        async with cond:
            cond.notify_all()

    async def _finalize_after(rnd):
        await asyncio.sleep(collect_window)    # backstop: void/aggregate once the upload window closes
        await _finalize(rnd)

    @app.get("/status")
    async def status(model_id: str = ""):
        async with cond:
            return {"mode": agg.mode(model_id, time.monotonic()),
                    "k_min": agg.k_min, "k_target": agg.k_target}

    @app.post("/contribute_dp")
    async def contribute_dp(req: Request):
        # Stream the DP blob to its own temp object and fold it per-module — never whole in RAM.
        stream = req.stream()
        hdr, prefix = await _read_header(stream)
        if hdr is None:
            raise HTTPException(400, "malformed contribution blob")
        model_id = hdr.get("__metadata__", {}).get("model_id", "")
        if not model_id:
            raise HTTPException(400, "model_id required")
        async with cond:
            pre = agg.dp_precheck(model_id, client_ip(req), time.monotonic())
            if pre == "ok":
                lock = finalize_locks.setdefault(model_id, asyncio.Lock())
        if pre != "ok":
            raise HTTPException(400, pre)
        round_id, slot = "dp-" + uuid.uuid4().hex, "u"
        writer = await run_in_threadpool(store.stage_writer, agg.datadir, round_id, slot)
        try:
            await _stage_stream(stream, prefix, writer)
        except Exception:
            await run_in_threadpool(writer.abort)
            await run_in_threadpool(store.stage_cleanup, agg.datadir, round_id)
            raise HTTPException(400, "upload failed")
        try:
            async with lock:                   # serialize G writes with same-model cohort finalizes
                res = await run_in_threadpool(agg.dp_fold, model_id, round_id, slot)
        finally:
            await run_in_threadpool(store.stage_cleanup, agg.datadir, round_id)
        async with cond:
            cond.notify_all()                  # a fresh global may now be servable
        if res != "ok":
            raise HTTPException(400, res)
        return {"status": "ok"}

    @app.post("/round/register")
    async def register(req: Request):
        body = await req.json()
        model_id, pubkey = body.get("model_id", ""), body.get("pubkey", "")
        if not model_id or not pubkey:
            raise HTTPException(400, "model_id and pubkey required")
        if agg.allowlist is not None and model_id not in agg.allowlist:
            raise HTTPException(403, "model_id not accepted by this federation")
        ip = client_ip(req)
        async with cond:
            rnd = agg.add_registrant(model_id, pubkey, ip, time.monotonic())
            agg.try_seal(rnd)                  # this arrival may itself complete the cohort
            cond.notify_all()
            while not (rnd.sealed or rnd.failed):
                remaining = register_window - (time.monotonic() - rnd.created)
                if remaining <= 0:
                    agg.try_seal(rnd, final=True)  # deadline: seal if ≥k_min, else FAIL
                    cond.notify_all()
                    break
                try:
                    await asyncio.wait_for(cond.wait(), remaining)
                except asyncio.TimeoutError:
                    pass
                agg.try_seal(rnd)
            if rnd.sealed:
                peers, round_id = rnd.sealed_peers, rnd.round_id
                if not rnd.finalize_scheduled:
                    rnd.finalize_scheduled = True
                    asyncio.create_task(_finalize_after(rnd))
        if rnd.sealed:
            return {"round_id": round_id, "peers": peers}  # client echoes round_id on /contribute
        raise HTTPException(503, "cohort below privacy minimum; skipping this round")

    @app.post("/contribute")
    async def contribute(req: Request):
        # Read just the safetensors header to learn round_id/compat/spec, then stream the rest straight
        # into a per-upload temp object — the whole masked vector is never resident.
        stream = req.stream()
        hdr, prefix = await _read_header(stream)
        if hdr is None or "masked" not in hdr:
            raise HTTPException(400, "malformed contribution blob")
        meta, m = hdr.get("__metadata__", {}), hdr["masked"]   # m = masked tensor's dtype + shape
        masked_len = 1
        for d in m.get("shape", []):
            masked_len *= d
        round_id = meta.get("round_id", "")
        async with cond:
            rnd, slot, res = agg.begin_stage(round_id, meta.get("compat", ""), meta.get("spec", "[]"),
                                             client_ip(req),
                                             masked_i64=(m.get("dtype") == "I64" and len(m.get("shape", [])) == 1),
                                             masked_len=masked_len)
        if res != "ok":
            raise HTTPException(400, res)
        writer = await run_in_threadpool(store.stage_writer, agg.datadir, round_id, slot)
        try:
            await _stage_stream(stream, prefix, writer)
        except Exception:
            await run_in_threadpool(writer.abort)
            async with cond:
                agg.unreserve(rnd)
            raise HTTPException(400, "upload failed")
        async with cond:
            agg.mark_received(rnd, slot)
            done = agg.complete(rnd)
        if done:
            await _finalize(rnd)               # all sealed members in early -> aggregate now, skip the U wait
        return {"status": "ok"}

    @app.get("/global")
    async def global_(model_id: str = "", since: str = ""):
        res = await run_in_threadpool(agg.serve_global, model_id, since)
        if res is None:
            return Response(status_code=204)
        chunks, version = res
        return StreamingResponse(chunks, media_type="application/octet-stream",
                                 headers={"X-Cursor": str(version)})

    return app


async def _read_header(stream):
    """Pull from `stream` until the safetensors header is complete; return (header_dict, buffered_bytes).
    The caller hands the buffered bytes to `_stage_stream` so nothing already read is lost."""
    buf = bytearray()
    async for chunk in stream:
        buf += chunk
        if len(buf) >= 8:
            n = struct.unpack("<Q", bytes(buf[:8]))[0]
            if len(buf) >= 8 + n:
                try:
                    return json.loads(bytes(buf[8 : 8 + n])), bytes(buf)
                except Exception:
                    return None, bytes(buf)
    return None, bytes(buf)


async def _stage_stream(stream, prefix: bytes, writer) -> None:
    """Stream the upload body into `writer`, batching so each off-thread store write moves ~8 MiB (so the
    body never sits whole in RAM and S3 parts stay above the 5 MiB minimum)."""
    batch = bytearray(prefix)
    async for chunk in stream:
        batch += chunk
        if len(batch) >= _STAGE_BATCH:
            await run_in_threadpool(writer.write, bytes(batch))
            batch = bytearray()
    if batch:
        await run_in_threadpool(writer.write, bytes(batch))
    await run_in_threadpool(writer.commit)


def _trusted(ip: str, nets) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in n for n in nets)
