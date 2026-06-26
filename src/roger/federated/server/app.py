"""app.py — FastAPI wire layer for the aggregation server.

Implements the three endpoints the client's transport.py already speaks (no client change):
  POST /round/register   {model_id, pubkey(hex)} -> {peers:[hex,...]}   (long-polls until the cohort
                         seals; 503 if the cohort stays below k_min so the client skips — returning an
                         empty peer set would make the client upload UNMASKED, leaking its ΔW).
  POST /contribute       octet-stream safetensors ({"masked": int64} + meta model_id/compat/spec) -> 200
  GET  /global?since=&model_id=  -> 204, or 200 octet-stream dense ΔW + X-Cursor header.

Async is essential: /round/register holds the connection open on an asyncio.Condition until its round
seals, which is the barrier that gives every cohort member the *same* frozen peer set (a prerequisite
for the pairwise masks to cancel). All round-state mutation happens in the synchronous Aggregator
methods with no await in between, so under the single event-loop thread they are atomic — the
Condition only coordinates who waits and who wakes.
"""
import asyncio, ipaddress, os, time

from fastapi import FastAPI, HTTPException, Request, Response

from roger.federated import delta
from roger.federated.server.aggregate import Aggregator


def _env_set(name: str) -> set | None:
    v = os.environ.get(name, "").strip()
    return {x for x in (s.strip() for s in v.split(",")) if x} or None


def create_app(aggregator: Aggregator | None = None) -> FastAPI:
    """Build the app around an Aggregator (injectable for tests). Config comes from env so the same
    image deploys anywhere: ROGER_AGG_* knobs + ROGER_SERVER_DATA."""
    agg = aggregator or Aggregator(
        datadir=os.environ.get("ROGER_SERVER_DATA", "./server-data"),
        k_min=int(os.environ.get("ROGER_AGG_KMIN", "2")),
        k_target=int(os.environ.get("ROGER_AGG_KTARGET", "4")),
        eta=float(os.environ.get("ROGER_AGG_ETA", "1.0")),
        clip_norm=float(os.environ.get("ROGER_AGG_CLIP", "1.0")),
        ip_binding=os.environ.get("ROGER_AGG_IPBIND", "1") != "0",
        allowlist=_env_set("ROGER_AGG_MODELS"),
    )
    register_window = float(os.environ.get("ROGER_AGG_W", "20"))   # < client 30s register timeout
    collect_window = float(os.environ.get("ROGER_AGG_U", "20"))    # how long to wait for all uploads
    # Proxies whose X-Forwarded-For we trust for the real client IP (Caddy in front). Default: local.
    trusted = [ipaddress.ip_network(c.strip()) for c in
               os.environ.get("ROGER_AGG_TRUSTED_PROXIES", "127.0.0.1/32,::1/128,10.0.0.0/8").split(",") if c.strip()]

    app = FastAPI()
    cond = asyncio.Condition()

    def client_ip(req: Request) -> str:
        peer = req.client.host if req.client else ""
        xff = req.headers.get("x-forwarded-for")
        if xff and _trusted(peer, trusted):
            return xff.split(",")[0].strip()   # leftmost = original client when the hop is trusted
        return peer

    async def _finalize_after(rnd):
        # Backstop: void/aggregate the round once the upload window closes, even if not all uploaded.
        await asyncio.sleep(collect_window)
        async with cond:
            agg.finalize(rnd)
            cond.notify_all()

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
            cond.notify_all()                  # wake peers so they re-check the (possibly new) seal
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
        blob = await req.body()
        try:
            tensors, meta = delta.from_bytes(blob)
            masked = tensors["masked"]
        except Exception:
            raise HTTPException(400, "malformed contribution blob")
        round_id = meta.get("round_id", "")
        async with cond:
            res = agg.submit(round_id, masked, meta.get("compat", ""), meta.get("spec", "[]"), client_ip(req))
            rnd = agg.collecting.get(round_id)  # fetch before finalize() pops it from collecting
            if res == "ok" and rnd is not None and rnd.received == len(rnd.sealed_peers):
                agg.finalize(rnd)               # all sealed members in early -> aggregate now, skip the U wait
            cond.notify_all()
        if res != "ok":
            raise HTTPException(400, res)
        return {"status": "ok"}

    @app.get("/global")
    async def global_(model_id: str = "", since: str = ""):
        res = agg.serve_global(model_id, since)
        if res is None:
            return Response(status_code=204)
        buf, version = res
        return Response(content=buf, media_type="application/octet-stream", headers={"X-Cursor": str(version)})

    return app


def _trusted(ip: str, nets) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in n for n in nets)
