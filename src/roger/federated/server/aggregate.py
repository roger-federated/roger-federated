"""aggregate.py — the secure-aggregation round lifecycle + FedAvg accumulation (server core).

This is the *synchronous* heart of the server: round state machine + the recovery/aggregation math.
All the asyncio long-poll/barrier coordination lives in `app.py`; keeping this layer free of awaits
means every method here runs to completion atomically under FastAPI's single event-loop thread, so no
locks are needed, and the whole thing is unit-testable without a server (see tests/test_server.py).

Round lifecycle. Each model_id has ONE round open for registration at a time; when it seals it gets a
unique round_id and moves aside to collect, freeing a fresh registering round — so multiple cohorts of
the same model collect concurrently (uploads self-route by the round_id the client echoes back).
  REGISTERING  clients POST /round/register; we collect X25519 pubkeys + source IPs.
  -> sealed    when len(registrants) >= K_target, or at the W deadline if >= K_min (else FAILED:
               a sub-K_min cohort would leak an individual ΔW, so we refuse rather than seal). On seal
               the round is assigned a round_id (returned to its members) and moved to `collecting`.
  COLLECTING   sealed members POST /contribute with their round_id; we sum the masked int64 vectors
               mod R into that round.
  -> finalized when all sealed members uploaded (early) or the U deadline passes. Masks cancel ONLY
               if every sealed member uploads (no Shamir recovery currently), so a short
               member count => VOID the round (global untouched). This is the all-or-nothing model.

What the server can and cannot defend against is documented in client.py / the federated-server-roadmap
memory: secure aggregation hides individual ΔW, so we can only bound the *aggregate* (norm-clip +
small cohorts + small η), never filter a single poisoned-but-well-formed upload. That needs ZK range
proofs (RoFL/EIFFeL/ELSA), deferred.
"""
import json, uuid

import torch

from roger.federated import secure_agg
from roger.federated.server import store


class Round:
    """Mutable state for one aggregation round. A class (not a dict) because the lifecycle flags and
    the running int64 sum are genuinely stateful and mutated in place across many calls."""
    def __init__(self, model_id: str, created: float):
        self.model_id = model_id
        self.created = created                  # time.monotonic() at first registration (for the W deadline)
        self.registrants: dict[str, str] = {}   # pubkey hex -> source IP (insertion order = arrival order)
        self.sealed = False
        self.failed = False                     # cohort too small, or a poisoned/mismatched contribution
        self.finalized = False
        self.finalize_scheduled = False         # app sets this so the U-timeout task is spawned once
        self.round_id: str | None = None         # assigned on seal; clients echo it on /contribute
        self.sealed_peers: list[str] | None = None
        # set by the first valid contribution; the cohort must agree on layout for masks to cancel
        self.compat: str | None = None
        self.spec: list | None = None           # [(module_key, shape_tuple), ...]
        self.length: int | None = None
        self.running_sum: torch.Tensor | None = None  # int64, mod R
        self.received = 0

    def registrant_ips(self) -> set:
        return set(self.registrants.values())


def _parse_spec(spec_json: str) -> tuple[list, int]:
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
    def __init__(self, datadir: str, *, k_min: int = 2, k_target: int = 4, eta: float = 1.0,
                 clip_norm: float = 1.0, ip_binding: bool = True, allowlist: set | None = None):
        self.datadir = datadir
        self.k_min = k_min
        self.k_target = k_target
        self.eta = eta                          # server learning rate: G <- G + η·mean(ΔW)
        # The aggregate L2 cap is always applied: an honest cohort sums to ≤ k·clip_norm (matches the
        # client's CLIP_NORM=1.0), so we clip ‖ΣΔW‖ to that. Must be > 0.
        self.clip_norm = clip_norm
        self.ip_binding = ip_binding
        self.allowlist = allowlist              # None = accept any model_id
        self.registering: dict[str, Round] = {} # model_id -> the round currently open for registration
        self.collecting: dict[str, Round] = {}  # round_id -> a sealed round awaiting contributions (many at once)
        self.globals: dict[str, tuple] = store.load_all(datadir)  # model_id -> (tensors, version)

    # --- registration / sealing -------------------------------------------------------------

    def add_registrant(self, model_id: str, pubkey: str, ip: str, now: float) -> Round:
        """Join this model_id's open registering round, opening a fresh one if none is registering
        (the previous one has sealed and moved to `collecting`). Never rejects: a client arriving while
        a cohort is busy collecting simply starts/joins the NEXT cohort. The W deadline is fired by the
        long-polling waiters in app.py via try_seal(final=True), so no lazy expiry is needed here."""
        rnd = self.registering.get(model_id)
        if rnd is None:                          # none open (sealed/failed rounds were removed on seal)
            rnd = Round(model_id, created=now)
            self.registering[model_id] = rnd
        rnd.registrants[pubkey] = ip
        return rnd

    def try_seal(self, rnd: Round, *, final: bool = False) -> None:
        """Seal on reaching k_target (immediate) or, at the W deadline (`final`), on reaching k_min.
        On seal: assign a round_id, move the round to `collecting`, and clear it from `registering` so
        the next arrival opens a fresh cohort. A `final` round below k_min FAILS — sealing it would
        hand the server a near-unmasked single contribution."""
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

    # --- contribution -----------------------------------------------------------------------

    def submit(self, round_id: str, masked: torch.Tensor, compat: str, spec_json: str, ip: str) -> str:
        """Add one masked int64 vector into the named round's running sum (mod R). The client echoes
        its round_id (from /round/register) so uploads self-route even when several cohorts of the same
        model collect concurrently. Returns "ok" or an error string; an error means the contribution is
        NOT counted, so the round voids at the U deadline (a malformed/garbage upload is
        indistinguishable from a dropout — accepted for now)."""
        rnd = self.collecting.get(round_id)
        if rnd is None or rnd.finalized:
            return "no active round"
        if self.ip_binding and ip not in rnd.registrant_ips():
            return "ip not in cohort"           # weak (NAT/open registration) but free; see security notes
        if rnd.received >= len(rnd.sealed_peers):
            return "round full"                 # can't identify double-submits w/o signatures; cap the count
        if masked.dtype != torch.int64 or masked.dim() != 1:
            return "bad tensor"
        spec, length = _parse_spec(spec_json)
        if masked.numel() != length:
            return "length mismatch"
        if rnd.compat is None:                  # first contribution fixes the cohort's layout
            rnd.compat, rnd.spec, rnd.length = compat, spec, length
            rnd.running_sum = torch.zeros(length, dtype=torch.int64)
        elif compat != rnd.compat or length != rnd.length:
            rnd.failed = True                   # mixed layouts can't cancel coordinate-wise -> void
            return "compat mismatch"
        if (masked < 0).any() or (masked >= secure_agg.R).any():
            return "out of range"
        rnd.running_sum = (rnd.running_sum + masked) % secure_agg.R
        rnd.received += 1
        return "ok"

    # --- finalization (masks cancel -> dequantize -> FedAvg into the global) -----------------

    def finalize(self, rnd: Round) -> None:
        """Idempotent. On a complete cohort: sum cancels the pairwise masks, dequantize to ΣΔW,
        norm-bound it, and fold η·mean into the cumulative global. Any short/failed round is voided
        (global untouched)."""
        if rnd.finalized:
            return
        rnd.finalized = True
        if rnd.round_id:
            self.collecting.pop(rnd.round_id, None)
        if rnd.failed or not rnd.sealed or rnd.sealed_peers is None:
            return
        k = len(rnd.sealed_peers)
        if rnd.received != k or rnd.running_sum is None or rnd.spec is None:
            return                              # dropout -> void (no Shamir recovery currently)
        delta_sum = secure_agg.dequantize(rnd.running_sum % secure_agg.R, rnd.spec)  # = ΣΔW_i
        if any(not torch.isfinite(t).all() for t in delta_sum.values()):
            return                              # reject NaN/Inf outright
        # Aggregate norm-bound: an honest cohort sums to ≤ k·clip_norm; clip the whole sum to that.
        total = float(torch.sqrt(sum((t.float() ** 2).sum() for t in delta_sum.values())))
        scale = (self.clip_norm * k) / (total + 1e-12) if total > self.clip_norm * k else 1.0
        tensors, version = self.globals.get(rnd.model_id, ({}, 0))
        tensors = dict(tensors)                 # copy so a partial failure can't corrupt the served global
        for key, t in delta_sum.items():        # G ← G + η · mean(ΔW) = G + η·scale·ΣΔW/k
            upd = (self.eta * scale / k) * t.float()
            tensors[key] = upd if key not in tensors else tensors[key] + upd
        version += 1
        self.globals[rnd.model_id] = (tensors, version)
        store.save_global(self.datadir, rnd.model_id, tensors, version)

    # --- broadcast --------------------------------------------------------------------------

    def serve_global(self, model_id: str, since: str) -> tuple[bytes, int] | None:
        """The cumulative dense global for `model_id`, or None when nothing newer than `since`
        (cursor == version). The blob is exactly what the client folds at load."""
        from roger.federated import delta            # local import: keep torch/delta off the hot import path
        entry = self.globals.get(model_id)
        if entry is None:
            return None
        tensors, version = entry
        if since == str(version):
            return None
        return delta.to_bytes(tensors, model_id), version

    def _clear_registering(self, rnd: Round) -> None:
        if self.registering.get(rnd.model_id) is rnd:
            del self.registering[rnd.model_id]
