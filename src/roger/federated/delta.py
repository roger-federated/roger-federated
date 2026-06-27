"""delta.py — the federated contribution as a weight-space LoRA delta.

A training round produces a fresh LoRA adapter whose effective update is ΔW = scaling·B@A per
target module. We *densify* those factors into the full weight-space ΔW and upload that (after
secure-aggregation masking, see secure_agg.py). Sharing dense ΔW — rather than the raw A,B —
keeps cross-client aggregation sound regardless of each user's r/targets (B@A is well-defined;
summing independent factors is not), at the cost of a larger upload.

The server sums the masked ΔW across the federation and broadcasts the full cumulative *dense*
global (ΔW₁+ΔW₂+…). The client folds it into the base weights at load time (`fold_into`, before bnb
quantization) — never storing a model and never altering the HF cache. Compatibility between members
is just "same base model" = identical per-module (out, in) weight shapes, captured by `compat_hash`.
"""
import hashlib, json, struct

import torch
from safetensors.torch import load as st_load, save as st_save


def _dp_noise(A, B, z: float, generator):
    """Perturb both LoRA factors before B@A (the bootstrap DP step). ΔW is rank-r, so noising the
    factors keeps the noise in that signal subspace instead of over all out·in dense coords — far less
    SNR loss. Both factors, else it's projectable out of one. Per-factor σ = z·rms(factor), so z is a
    relative multiplier rather than an absolute tied to a model's weight scale."""
    def n(t):
        sigma = z * t.pow(2).mean().sqrt()      # rms of this (pre-clip) factor
        return t + torch.randn(t.shape, generator=generator) * sigma
    return n(A), n(B)


def densify(delta: dict, *, noise_z: float = 0.0, generator=None) -> dict:
    """{module: dense ΔW = scaling·(B@A)} from a round's PEFT factors (`delta["weights"]` = peft state
    dict, `["scaling"]` = alpha/r). noise_z>0 adds DP-bootstrap factor noise (see `_dp_noise`)."""
    sd, scaling = delta["weights"], float(delta["scaling"])
    out = {}
    for key, A in sd.items():
        if not key.endswith(".lora_A.weight"):
            continue
        mod = key[: -len(".lora_A.weight")]
        B   = sd[mod + ".lora_B.weight"]
        Af, Bf = A.float(), B.float()
        if noise_z:
            Af, Bf = _dp_noise(Af, Bf, noise_z, generator)
        out[mod] = (scaling * (Bf @ Af)).to(B.dtype)
    return out


def compat_hash(tensors: dict) -> str:
    """Stable digest of the base architecture this delta targets: sorted module → (out, in). Dense
    ΔW carries (out, in) directly; LoRA factors give out from lora_B[:,0], in from lora_A[0,:], so a
    dense upload and the re-factored broadcast of the same base hash identically."""
    shapes = {}
    for key, t in tensors.items():
        if key.endswith(".lora_A.weight"):
            shapes.setdefault(key[: -len(".lora_A.weight")], [None, None])[1] = t.shape[1]   # in
        elif key.endswith(".lora_B.weight"):
            shapes.setdefault(key[: -len(".lora_B.weight")], [None, None])[0] = t.shape[0]   # out
        else:                                              # dense ΔW [out, in]
            shapes[key] = [t.shape[0], t.shape[1]]
    blob = ";".join(f"{m}:{s[0]}x{s[1]}" for m, s in sorted(shapes.items()))
    return hashlib.sha1(blob.encode()).hexdigest()


def _read_metadata(buf: bytes) -> dict:
    # safetensors layout: u64 LE header length, then the JSON header (whose "__metadata__" holds our
    # str→str fields). load() drops it, so parse the header directly rather than round-tripping a file.
    n = struct.unpack("<Q", buf[:8])[0]
    return json.loads(buf[8 : 8 + n]).get("__metadata__", {})


def to_bytes(tensors: dict, model_id: str) -> bytes:
    """Serialize a tensor dict (dense ΔW for upload, or factors for a broadcast) with model_id +
    base-compat hash in the metadata."""
    return st_save(tensors, metadata={"model_id": model_id, "compat": compat_hash(tensors)})


def from_bytes(buf: bytes) -> tuple[dict, dict]:
    return st_load(buf), _read_metadata(buf)


def _base_key(module_path: str) -> str:
    """A densified ΔW is keyed by the PEFT module path (`base_model.model.<base path>`); the bare base
    model exposes it at `<base path>`. Strip the PEFT wrapper prefix to reach the real submodule."""
    return module_path[len("base_model.model.") :] if module_path.startswith("base_model.model.") else module_path


def fold_into(model, deltas: dict) -> int:
    """Add each dense ΔW into the matching base weight *in place* (call on the bf16 model, before
    quantization). Returns how many modules were folded; warns and skips any whose submodule is
    missing or whose shape disagrees, so a wrong-base broadcast can't silently corrupt weights."""
    import warnings
    folded = 0
    for module_path, dW in deltas.items():
        try:
            w = model.get_submodule(_base_key(module_path)).weight
        except AttributeError:
            warnings.warn(f"federated: no weight for {module_path}; skipping. Expect decreased performance.")
            continue
        if tuple(w.shape) != tuple(dW.shape):
            warnings.warn(f"federated: shape mismatch at {module_path} "
                          f"({tuple(w.shape)} vs {tuple(dW.shape)}); skipping. Expect decreased performance.")
            continue
        w.data += dW.to(w.dtype, copy=False).to(w.device)
        folded += 1
    return folded
