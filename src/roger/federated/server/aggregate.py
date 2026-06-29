"""aggregate.py — secure-aggregation round lifecycle + FedAvg accumulation (server core).

Lock-held methods (register, seal, stage bookkeeping, claim_finalize) run with no await, so they are
atomic under the single event-loop thread; the I/O-heavy `run_finalize` runs OFF it in a threadpool,
touching only a round already pulled from `collecting` + storage, so it can't race the lock-held state.

Lifecycle (per model_id, one registering round at a time; on seal it gets a round_id and moves aside,
so cohorts collect concurrently, self-routed by round_id):
  REGISTERING  /round/register collects X25519 pubkeys + IPs.
  -> sealed    at K_target, or at the W deadline if >= K_min (else FAILED: sub-K_min leaks a lone ΔW).
  COLLECTING   /contribute streams each masked upload to its OWN temp object (never a RAM running-sum).
  -> finalized all uploaded, or U deadline: sum staged slices per module (masks cancel), fold into the
               global, voiding if ‖ΣΔW‖ exceeds k·clip. Masks cancel only if EVERY member uploaded.

Staging (not a RAM running-sum) because a masked vector spans the full dense basis: `8·P_target` bytes
resident for the whole window × concurrent cohorts overruns serverless tiers; per-module keeps RAM ~one
matrix. Bootstrap (sparse `mode()`): one unmasked DP ΔW to /contribute_dp, `dp_fold` (k=1) — the same
per-module streamed fold, just no cohort.
"""
import json, uuid

import torch

from roger.federated import secure_agg
from roger.federated.server import store


class Round:
    """Mutable state for one aggregation round. A class (not a dict) because the lifecycle flags and the
    staged-upload bookkeeping are genuinely stateful and mutated in place across many calls."""
    def __init__(self, model_id: str, created: float):
        self.model_id = model_id
        self.created = created                  # time.monotonic() at first registration (W deadline)
        self.registrants: dict[str, str] = {}   # pubkey hex -> source IP (insertion order = arrival)
        self.sealed = False
        self.failed = False                     # cohort too small, or a mismatched contribution
        self.finalized = False
        self.finalize_scheduled = False         # app spawns the U-timeout task once
        self.round_id: str | None = None         # assigned on seal; clients echo it on /contribute
        self.sealed_peers: list[str] | None = None
        # set by the first valid contribution; the cohort must agree on layout for masks to cancel
        self.compat: str | None = None
        self.spec: list | None = None           # [(module_key, shape_tuple), ...]
        self.length: int | None = None
        self.staged: set = set()                # slot ids of fully-uploaded temp objects
        self.inflight = 0                       # reservations whose upload is still streaming
        self.received = 0

    def registrant_ips(self) -> set:
        return set(self.registrants.values())


def _parse_spec(spec_json: str) -> tuple:
    """JSON [[key,[shape...]],...] -> ([(key, shape_tuple)], total_flat_length)."""
    spec, length = [], 0
    for key, shape in json.loads(spec_json):
        shape = tuple(shape)
        spec.append((key, shape))
        n = 1
        for d in shape:
            n *= d
        length += n
    return spec, length


class Aggregator:
    def __init__(self, datadir: str, *, k_min: int = 3, k_target: int = 5, eta: float = 1.0,
                 eta_boot: float | None = None, clip_norm: float = 1.0, ip_binding: bool = True,
                 allowlist: set | None = None, busy_threshold: int | None = None,
                 busy_window: float = 180.0):
        self.datadir = datadir
        self.k_min = k_min
        self.k_target = k_target
        self.eta = eta                          # server learning rate: G <- G + η·mean(ΔW)
        self.eta_boot = eta if eta_boot is None else eta_boot  # rate for a single async DP upload
        self.clip_norm = clip_norm              # aggregate L2 cap (honest cohort sums to ≤ k·clip_norm)
        self.ip_binding = ip_binding
        self.allowlist = allowlist              # None = accept any model_id
        self.busy_threshold = k_target if busy_threshold is None else busy_threshold
        self.busy_window = busy_window
        self.recent: dict[str, dict[str, float]] = {}  # model_id -> {ip: last-seen monotonic time}
        self.registering: dict[str, Round] = {} # model_id -> the round open for registration
        self.collecting: dict[str, Round] = {}  # round_id -> a sealed round awaiting uploads (many at once)

    # --- mode / density ---------------------------------------------------------------------

    def mode(self, model_id: str, now: float) -> str:
        """"busy" (secure-agg cohorts) once >= busy_threshold distinct contributors fall inside the
        busy_window, else "bootstrap" (async DP). Prunes the window in passing."""
        seen = self.recent.get(model_id)
        if not seen:
            return "bootstrap"
        cutoff = now - self.busy_window
        live = {ip: t for ip, t in seen.items() if t >= cutoff}
        self.recent[model_id] = live
        return "busy" if len(live) >= self.busy_threshold else "bootstrap"

    def _note_contributor(self, model_id: str, ip: str, now: float) -> None:
        self.recent.setdefault(model_id, {})[ip] = now

    # --- registration / sealing -------------------------------------------------------------

    def add_registrant(self, model_id: str, pubkey: str, ip: str, now: float) -> Round:
        """Join this model_id's open registering round, opening a fresh one if none is registering."""
        self._note_contributor(model_id, ip, now)
        rnd = self.registering.get(model_id)
        if rnd is None:
            rnd = Round(model_id, created=now)
            self.registering[model_id] = rnd
        rnd.registrants[pubkey] = ip
        return rnd

    def try_seal(self, rnd: Round, *, final: bool = False) -> None:
        """Seal at k_target (immediate) or, at the W deadline (`final`), at k_min. A `final` round below
        k_min FAILS — sealing it would hand the server a near-unmasked single contribution."""
        if rnd.sealed or rnd.failed:
            return
        n = len(rnd.registrants)
        if n >= self.k_target or (final and n >= self.k_min):
            rnd.sealed = True
            rnd.sealed_peers = list(rnd.registrants.keys())
            rnd.round_id = uuid.uuid4().hex
            self.collecting[rnd.round_id] = rnd
            self._clear_registering(rnd)
        elif final:
            rnd.failed = True
            self._clear_registering(rnd)

    # --- contribution staging (lock-held bookkeeping; the I/O is in app.py) ------------------

    def begin_stage(self, round_id: str, compat: str, spec_json: str, ip: str, *,
                    masked_i64: bool = True, masked_len: int | None = None) -> tuple:
        """Validate an incoming upload and reserve a slot. Returns (round, slot_id, "ok") or
        (None, None, error). The caller streams the body to `store.stage_writer(.., round_id, slot)`,
        then calls mark_received (success) or unreserve (failure). Reserving under the lock caps the
        cohort at k even with concurrent uploads. masked_i64/masked_len come from the upload's
        safetensors header (we never decode the whole tensor here)."""
        rnd = self.collecting.get(round_id)
        if rnd is None or rnd.finalized:
            return None, None, "no active round"
        if self.ip_binding and ip not in rnd.registrant_ips():
            return None, None, "ip not in cohort"   # weak (NAT/open registration) but free
        if not masked_i64:
            return None, None, "bad tensor"          # masked must be a 1-D int64 residue vector
        if len(rnd.staged) + rnd.inflight >= len(rnd.sealed_peers):
            return None, None, "round full"          # can't dedup peers w/o signatures; cap the count
        spec, length = _parse_spec(spec_json)
        if masked_len is not None and masked_len != length:
            return None, None, "length mismatch"
        if rnd.compat is None:                        # first contribution fixes the cohort's layout
            rnd.compat, rnd.spec, rnd.length = compat, spec, length
        elif compat != rnd.compat or length != rnd.length:
            rnd.failed = True                         # mixed layouts can't cancel coordinate-wise -> void
            return None, None, "compat mismatch"
        rnd.inflight += 1
        return rnd, uuid.uuid4().hex, "ok"

    def mark_received(self, rnd: Round, slot: str) -> None:
        rnd.inflight -= 1
        rnd.staged.add(slot)
        rnd.received = len(rnd.staged)

    def unreserve(self, rnd: Round) -> None:
        rnd.inflight -= 1

    def complete(self, rnd: Round) -> bool:
        return rnd.received == len(rnd.sealed_peers)

    # --- bootstrap DP fold (per-module streamed, k=1) ---------------------------------------

    def dp_precheck(self, model_id: str, ip: str, now: float) -> str:
        """Cheap + lock-held: allowlist + count the uploader toward the busy threshold."""
        if self.allowlist is not None and model_id not in self.allowlist:
            return "model_id not accepted"
        self._note_contributor(model_id, ip, now)
        return "ok"

    def dp_fold(self, model_id: str, round_id: str, slot: str) -> str:
        """I/O-bound (run off-thread): per-module fold of ONE staged unmasked dense ΔW (k=1). Reads the
        upload module-by-module from its staged temp object, so nothing is whole in RAM."""
        up = store.open_staged_reader(self.datadir, round_id, slot)
        if up is None or not up.keys:
            return "empty delta"
        old_reader = store.open_global_reader(self.datadir, model_id)
        old = old_reader.keys if old_reader is not None else {}
        for key, (_, shape) in up.keys.items():
            if len(shape) != 2:
                return "bad delta"               # dense ΔW is [out, in]
            if key in old and tuple(old[key][1]) != tuple(shape):
                return "shape mismatch"          # would corrupt the global's per-coord sum
        delta_shape = {key: shape for key, (_, shape) in up.keys.items()}
        return self._fold_streamed(model_id, delta_shape, lambda key, shape: up.read(key), k=1, eta=self.eta_boot)

    def submit_dp(self, model_id: str, deltas: dict, ip: str, now: float) -> str:
        """Sync bootstrap submit (tests / direct use): stage the ΔW then per-module fold. The app instead
        streams the upload straight into the temp object. "ok" or an error/void reason."""
        pre = self.dp_precheck(model_id, ip, now)
        if pre != "ok":
            return pre
        if not deltas:
            return "empty delta"
        from roger.federated import delta as delta_mod
        round_id, slot = "dp-" + uuid.uuid4().hex, "u"
        w = store.stage_writer(self.datadir, round_id, slot)
        w.write(delta_mod.to_bytes(deltas, model_id))
        w.commit()
        try:
            return self.dp_fold(model_id, round_id, slot)
        finally:
            store.stage_cleanup(self.datadir, round_id)

    # --- cohort finalization (claim under lock, aggregate off-thread) ------------------------

    def claim_finalize(self, rnd: Round) -> str:
        """Atomically claim a round for finalization (idempotent). Returns "aggregate" (this caller owns
        a complete cohort), "void" (this caller owns an incomplete/failed round — cleanup only), or
        "done" (already claimed). The heavy work then runs in run_finalize off the event loop."""
        if rnd.finalized:
            return "done"
        rnd.finalized = True
        if rnd.round_id:
            self.collecting.pop(rnd.round_id, None)
        if (rnd.failed or not rnd.sealed or rnd.sealed_peers is None
                or rnd.received != len(rnd.sealed_peers) or rnd.spec is None):
            return "void"
        return "aggregate"

    def run_finalize(self, rnd: Round, status: str) -> bool:
        """I/O-bound; run in a threadpool (no event-loop lock). Aggregate a complete cohort into the
        global, then always delete the round's temp objects. Returns whether the global was updated."""
        updated = False
        try:
            if status == "aggregate":
                updated = self._aggregate_streamed(rnd, len(rnd.sealed_peers))
        except Exception:
            updated = False                      # writer aborted inside; global untouched -> void
        finally:
            if rnd.round_id:
                store.stage_cleanup(self.datadir, rnd.round_id)
        return updated

    def _module_sum(self, rnd: Round, slots: list, data_starts: dict, key: str, shape: tuple,
                    moff: int, n: int) -> torch.Tensor:
        """ΣΔW for one module: sum each slot's int64 slice mod R (masks cancel per coord), dequantize."""
        acc = None
        for slot in slots:
            start = data_starts[slot] + moff * 8         # staged "masked" is int64 (8 B/elem)
            s = torch.frombuffer(bytearray(store.stage_read_slice(self.datadir, rnd.round_id, slot, start, n * 8)),
                                 dtype=torch.int64)
            acc = s.clone() if acc is None else (acc + s) % secure_agg.R
        return secure_agg.dequantize(acc % secure_agg.R, [(key, tuple(shape))])[key]

    def _aggregate_streamed(self, rnd: Round, k: int) -> bool:
        """Cohort finalize: reconstruct ΣΔW per module from the staged slices, then stream-fold it."""
        slots = list(rnd.staged)
        data_starts = {}
        for slot in slots:
            ds, length = store.stage_header(self.datadir, rnd.round_id, slot)
            if length != rnd.length:
                return False                             # malformed staged object -> void
            data_starts[slot] = ds

        offsets, off = {}, 0                             # key -> (flat-offset, flat-len), spec order
        for key, shape in rnd.spec:
            n = 1
            for d in shape:
                n *= d
            offsets[key] = (off, n)
            off += n
        delta_shape = {key: shape for key, shape in rnd.spec}
        module = lambda key, shape: self._module_sum(rnd, slots, data_starts, key, shape, *offsets[key])
        return self._fold_streamed(rnd.model_id, delta_shape, module, k=k, eta=self.eta) == "ok"

    def _fold_streamed(self, model_id: str, delta_shape: dict, module_delta, *, k: int, eta: float) -> str:
        """Fold η·mean(ΔW) into the global one module at a time (peak RAM ~one matrix), in a SINGLE pass:
        stream oldG + (η/k)·ΣΔW while measuring ‖ΣΔW‖, then commit iff within k·clip. An honest clipped
        cohort always is (triangle ineq); an over-norm aggregate only comes from a client that skipped
        its clip, so we VOID it (consistent with dropout-void). `module_delta(key, shape)` yields ΣΔW for
        one module (cohort: summed staged slices; bootstrap: the uploaded tensor)."""
        reader = store.open_global_reader(self.datadir, model_id)
        old = dict(reader.keys) if reader is not None else {}
        order = sorted(set(old) | set(delta_shape))
        specs = [(key, "F32", delta_shape[key] if key in delta_shape else old[key][1]) for key in order]

        writer = store.global_writer(self.datadir, model_id, specs)
        sq, coef = 0.0, eta / k
        try:
            for key in order:
                base = reader.read(key).float() if (reader is not None and key in old) else None
                if key in delta_shape:
                    dW = module_delta(key, delta_shape[key]).float()
                    if not torch.isfinite(dW).all():
                        writer.abort()
                        return "non-finite delta"        # void: global untouched
                    sq += float((dW ** 2).sum())
                    val = coef * dW if base is None else base + coef * dW
                else:
                    val = base                           # old-only module -> carried over unchanged
                writer.write(val.contiguous().to(torch.float32).numpy().tobytes())
        except Exception:
            writer.abort()
            raise
        if sq ** 0.5 > self.clip_norm * k:               # over the honest bound -> void, don't fold abuse
            writer.abort()
            return "norm exceeded"
        writer.commit(store.load_version(self.datadir, model_id) + 1)
        return "ok"

    # --- broadcast --------------------------------------------------------------------------

    def serve_global(self, model_id: str, since: str):
        """(chunk_iter, version) for the cumulative global, or None when nothing newer than `since`
        (cursor == version) or no global yet. The stream is exactly what the client folds."""
        version = store.load_version(self.datadir, model_id)
        if version == 0 or since == str(version):
            return None
        res = store.open_global_stream(self.datadir, model_id)
        if res is None:
            return None
        return res[0], version

    def _clear_registering(self, rnd: Round) -> None:
        if self.registering.get(rnd.model_id) is rnd:
            del self.registering[rnd.model_id]
