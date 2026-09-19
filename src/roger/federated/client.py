"""client.py — the federated upload, from the training round's point of view.

One entry point: `contribute_factor(url, …)` sends one round's LoRA-factor update to ONE federation,
in the regime that federation's `/status` reported — secure-agg masked (busy) or DP-noised and
unmasked (bootstrap). The pull half needs no torch and lives in runtime/adapter.py, which materialises
the pulled global as the runtime's own adapter.
"""
import json

import torch
from safetensors.torch import save as st_save

from roger.federated import delta as delta_mod, secure_agg, transport

# Per-client L2 bound on the shared update — the clipping step of DP-FedAvg (McMahan et al. 2018), here
# applied within secure aggregation in the spirit of cpSGD (Agarwal et al. 2018). It is NOT
# user-configurable on purpose: a clip the contributor chooses (or skips) is not a security boundary
# — a malicious client just removes it. It's best-effort here; the *authoritative* per-client norm
# bound must be enforced server-side, which under masking needs a zero-knowledge range proof
# (RoFL / EIFFeL / ELSA) since the server never sees an individual unmasked update. See
# federated_server_requirements.
CLIP_NORM = 1.0

# Bootstrap factor-noise multiplier (σ = z·rms(Δ)). Obfuscation, not a budgeted (ε,δ) guarantee — a
# starting knob to tune against utility. The frozen factor is public and the map Δ ↦ Δ·A is linear, so
# the weight-space noise is exactly Gaussian; z=0.3 → 18% noise power (~42% amplitude) in the factor.
DP_Z = 0.3


def _pack(masked: torch.Tensor, spec: list, compat: str, model_id: str, round_id: str,
          token: str, stamp: dict) -> bytes:
    # Carry the layout (spec) + base hash so the server can rebuild, place, and check each Δ, the
    # round_id from registration so the server routes this upload to the cohort we masked against, and
    # the token proving we're the registrant who masked against that cohort's peer set. `stamp` = the
    # factor contract's {base, epoch} (see delta.py).
    spec_json = [[k, list(shape)] for k, shape in spec]
    return st_save({"masked": masked},
                   metadata={"model_id": model_id, "compat": compat,
                             "spec": json.dumps(spec_json), "round_id": round_id, "token": token,
                             **stamp})


def _clip_factor(update: dict, max_norm: float) -> dict:
    # Best-effort bound on the update's global L2 norm in factor space: the server voids a cohort whose
    # ‖ΣΔ‖ exceeds k·clip, so an honest member keeps its own ‖Δ‖ within the clip (see CLIP_NORM).
    total = float(torch.sqrt(sum((t.float() ** 2).sum() for t in update.values())))
    if total <= max_norm:
        return update
    return {k: v.float() * (max_norm / (total + 1e-12)) for k, v in update.items()}


def contribute_factor(url: str, update: dict, base: dict, epoch: int, model_id: str,
                      mode: str = "busy") -> bool:
    """Upload one round's factor update {`<module>.lora_B.weight` (or A): Δ} to ONE federation, in the
    regime `mode` its /status reported. Under the factor contract a member contributes to exactly one
    federation — the one whose global it trained against — since Δ is only meaningful next to that
    federation's frozen factor. Returns whether the upload was accepted; the caller keeps its data
    otherwise.
    bootstrap: Gaussian noise straight on the factor (σ = DP_Z·rms(Δ)); the frozen factor is public and
      the map Δ ↦ Δ·A linear, so the weight-space noise is exactly Gaussian.
    busy: quantize + mask against the sealed cohort; dropped if the cohort sealed in a later epoch,
      since the server would void it anyway."""
    update = _clip_factor(update, CLIP_NORM)
    stamp = {"base": delta_mod.base_to_json(base), "epoch": str(epoch)}
    if mode == "bootstrap":
        noisy = {}
        for k, v in update.items():
            v = v.float()
            noisy[k] = v + torch.randn(v.shape) * (DP_Z * v.pow(2).mean().sqrt())
        return transport.contribute_dp(url, delta_mod.to_bytes(noisy, model_id, stamp)) == "ok"
    q, spec = secure_agg.quantize(update)
    priv, pub = secure_agg.gen_keypair()
    res = transport.register_and_peers(url, pub, model_id)
    if res is None:                              # unreachable/sub-quorum: never upload an unmaskable payload
        return False
    round_id, token, peers, sealed_epoch = res
    if sealed_epoch is not None and int(sealed_epoch) != epoch:
        return False                             # trained against a factor the federation moved past
    compat = delta_mod.compat_from_shapes(base)
    blob = _pack(secure_agg.mask(q, priv, peers), spec, compat, model_id, round_id, token, stamp)
    return transport.contribute(url, blob) == "ok"
