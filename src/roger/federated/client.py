"""client.py — orchestrates a federated round from the CLI's point of view.

Entry points the CLI calls:
  maybe_daily_pull(cfg)         — first startup of a new UTC day: download + persist the global blob.
  pending_globals(cfg)          — at model load: the dense ΔW to fold into the base (summed across
                                  federations), or None.
  contribute_delta(delta, cfg)  — after a training round: secure-agg masked upload (busy) or, while
                                  sparse, an async DP-noised unmasked upload (bootstrap); server picks.

Plus is_leeching(cfg) for the startup reprimand. All of this is a no-op when no federation is
configured, so the rest of the app is unaffected when sharing is off.
"""
import json
from datetime import datetime, timezone

import torch
from safetensors.torch import save as st_save

from roger.federated import CLIENT_VERSION, delta as delta_mod, secure_agg, transport

# Per-client L2 bound on the shared ΔW — the clipping step of DP-FedAvg (McMahan et al. 2018), here
# applied within secure aggregation in the spirit of cpSGD (Agarwal et al. 2018). It is NOT
# user-configurable on purpose: a clip the contributor chooses (or skips) is not a security boundary
# — a malicious client just removes it. It's best-effort here; the *authoritative* per-client norm
# bound must be enforced server-side, which under masking needs a zero-knowledge range proof
# (RoFL / EIFFeL / ELSA) since the server never sees an individual unmasked ΔW. See
# federated_server_requirements.
CLIP_NORM = 1.0

# Bootstrap factor-noise multiplier (delta._dp_noise: σ = z·rms(factor)). Obfuscation, not a budgeted
# (ε,δ) guarantee — a starting knob to tune against utility. noise/signal power in the densified ΔW
# ≈ 2z², shape-independent (same ratio for every target module regardless of rank/dims), so one
# global z suffices. z=0.3 → 18% noise power (~42% amplitude); z=0.5 was ~50% power (~71% amplitude).
DP_Z = 0.3


def is_leeching(cfg: dict) -> bool:
    """Configured into a federation but refusing to contribute = leeching (still pulls the global)."""
    return bool(cfg.get("federations")) and not cfg.get("contribute", True)


def probe_federations(cfg: dict) -> dict[str, dict]:
    """One /status GET per configured federation (fail-soft to {}), keyed by URL. A single pass yields
    every verdict the CLI needs — unsupported model, outdated client, update-available — so probing the
    servers three times (startup) is avoided; the helpers below just read this dict."""
    model_id = cfg.get("model_id", "")
    return {url: transport.federation_status(url, model_id)
            for url in (cfg.get("federations") or [])}


def unsupported_urls(statuses: dict) -> list[str]:
    """Feds whose allowlist excludes the model (mode "unsupported"): they won't take this session's
    gradient or serve updates. Fail-soft {} reads as "busy", so only a *definite* rejection is flagged
    and a network hiccup never false-warns."""
    return [url for url, s in statuses.items() if s.get("mode", "busy") == "unsupported"]


def outdated_urls(statuses: dict) -> list[str]:
    """Feds whose min_client exceeds this build: our contribution would be rejected/misaggregated, so we
    skip them exactly like an unsupported model (keeping the runs). Absent min_client (0, a pre-version
    or unreachable server) never flags — same fail-soft as unsupported_urls."""
    return [url for url, s in statuses.items() if CLIENT_VERSION < int(s.get("min_client", 0) or 0)]


def newest_client(statuses: dict) -> int:
    """Highest latest_client any fed advertises above our version (advisory update notice), else 0."""
    latest = max((int(s.get("latest_client", 0) or 0) for s in statuses.values()), default=0)
    return latest if latest > CLIENT_VERSION else 0


def should_train(cfg: dict) -> bool:
    """Local training only earns its keep when there's a federation to send the ΔW to and the user
    is contributing — there is no local-apply path, so otherwise the round would be discarded."""
    return bool(cfg.get("federations")) and cfg.get("contribute", True)


def _clip(dense: dict, max_norm: float) -> dict:
    """Best-effort bound on the shared ΔW's global L2 norm (caps an honest client's influence; the
    authoritative bound is server-side, see CLIP_NORM). Returns dense unchanged when within bounds."""
    if not max_norm:
        return dense
    total = float(torch.sqrt(sum((t.float() ** 2).sum() for t in dense.values())))
    if total <= max_norm:
        return dense
    s = max_norm / (total + 1e-12)
    return {k: (v.float() * s).to(v.dtype) for k, v in dense.items()}


def _pack(masked: torch.Tensor, spec: list, compat: str, model_id: str, round_id: str,
          token: str) -> bytes:
    # Carry the layout (spec) + base hash so the server can rebuild, place, and check each ΔW, the
    # round_id from registration so the server routes this upload to the cohort we masked against, and
    # the token proving we're the registrant who masked against that cohort's peer set.
    spec_json = [[k, list(shape)] for k, shape in spec]
    return st_save({"masked": masked},
                   metadata={"model_id": model_id, "compat": compat,
                             "spec": json.dumps(spec_json), "round_id": round_id, "token": token})


def contribute_delta(delta: dict, cfg: dict) -> bool:
    """Upload the round's ΔW to each federation in the regime the server reports: the secure-agg cohort
    path (busy), or the cohort-free async DP path (bootstrap) while the federation is too sparse to seal.
    No-op for a leech / empty feds / empty delta. Returns whether *any* upload was accepted; the caller
    keeps the runs for retry otherwise."""
    feds = cfg.get("federations") or []
    if not cfg.get("contribute", True) or not feds or not delta:
        return False
    model_id = delta["model_id"]
    secure = None                               # (compat, q, spec) for the busy path; built once on demand
    accepted = False
    for url in feds:
        st = transport.federation_status(url, model_id)
        mode = st.get("mode", "busy")
        if mode == "unsupported" or CLIENT_VERSION < int(st.get("min_client", 0) or 0):
            continue                            # model not accepted here, or our client is too old: skip
        if mode == "bootstrap":
            # Cohort-free: noise the factors, clip, upload unmasked. Re-noised per federation.
            noisy = _clip(delta_mod.densify(delta, noise_z=DP_Z), CLIP_NORM)
            if transport.contribute_dp(url, delta_mod.to_bytes(noisy, model_id)) == "ok":
                accepted = True
            continue
        # Busy: mask against the sealed cohort. The dense ΔW is identical across feds, so quantize once.
        if secure is None:
            dense = _clip(delta_mod.densify(delta), CLIP_NORM)
            q, spec = secure_agg.quantize(dense)
            secure = (delta_mod.compat_hash(dense), q, spec)
        compat, q, spec = secure
        priv, pub = secure_agg.gen_keypair()
        res = transport.register_and_peers(url, pub, model_id)
        if res is None:                         # unreachable/sub-quorum: don't upload an unmaskable payload
            continue
        round_id, token, peers = res
        masked = secure_agg.mask(q, priv, peers)
        # "ok" = received into a collecting cohort, not that the round finalized (no finalization signal).
        if transport.contribute(url, _pack(masked, spec, compat, model_id, round_id, token)) == "ok":
            accepted = True
    return accepted


def maybe_daily_pull(cfg: dict) -> bool:
    """Once per UTC day per federation, download the full cumulative global ΔW and persist the blob
    (re-folded at every model load by `pending_globals`; no model is touched here). Returns whether a
    fresh global was fetched. Stamps the date even on a no-op so we don't re-poll all day."""
    feds = cfg.get("federations") or []
    if not feds:
        return False
    today, fetched = datetime.now(timezone.utc).date().isoformat(), False
    for url in feds:
        st = transport.load_state(url)
        if st.get("last_sync") == today:
            continue
        res = transport.pull(url, st.get("cursor"), cfg.get("model_id", ""))
        if res is not None:
            buf, cursor = res
            transport.save_global(url, buf)
            st["cursor"], fetched = cursor, True
        st["last_sync"] = today
        transport.save_state(url, st)
    return fetched


def pending_globals(cfg: dict) -> dict | None:
    """The dense ΔW to fold into the base at load time: the persisted cumulative global of each
    federation, summed (default is a single federation). None when nothing is persisted yet."""
    feds = cfg.get("federations") or []
    summed: dict = {}
    for url in feds:
        blob = transport.load_global(url)
        if blob is None:
            continue
        tensors, _meta = delta_mod.from_bytes(blob)
        for k, v in tensors.items():
            summed[k] = v if k not in summed else summed[k] + v
    return summed or None
