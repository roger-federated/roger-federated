"""delta.py — the federated update as LoRA factors: the wire contract shared with the server.

Mirrors `roger_server/delta.py` in the roger-server repo BY HAND. Both sides exchange **LoRA factors
only**, never a dense ΔW: the global IS the adapter the runtime attaches (runtime/adapter.py), stored and
broadcast as the PEFT pair `<module>.lora_A.weight` [r, in] / `<module>.lora_B.weight` [out, r] (F32,
`scaling` = "1" in the metadata since the scale is already folded into the factors).

One factor at a time (RoLoRA; Chen et al. 2024, arXiv:2410.07739). Secure aggregation only ever hands
the server the SUM of the cohort's uploads, and factors do not sum: (ΣB)(ΣA) leaves the cross terms
B_i A_j behind. So a federation alternates epochs: in a B-epoch A is frozen and identical for everyone,
members train B only and upload ΔB, and Σ(ΔB_i)·A = Σ(ΔB_i·A) exactly; an A-epoch mirrors it. Epoch 0
trains B (LoRA starts at B=0, so A's gradient would be zero). `/status` advertises `epoch`/`phase`/`rank`,
and every upload is stamped with the epoch it trained in plus the base shape map (`base`) — an upload
from a past epoch is voided server-side.

Rank is per module but STATIC (`rank_for`: a pure function of the module's shape and the federation's
cap), and a cold federation's frozen A is DERIVED (`init_A`), never transmitted — so a client with no
global to pull yet trains against the exact A the server will fold into. Changing either rule, the
metadata keys, or `compat_hash` is a breaking change on both sides.

The legacy in-process CLI still folds a (densified) global into the base weights at load time
(`densify` + `fold_into`); its dense upload path no longer matches the server.
"""
import hashlib, json, struct

import torch
from safetensors.torch import load as st_load, save as st_save

LORA_A = ".lora_A.weight"
LORA_B = ".lora_B.weight"
SUFFIX = {"A": LORA_A, "B": LORA_B}          # phase -> the factor key that phase trains

# Rank rule (contract): a module's rank is ~1/128 of its full rank, floored so even a small matrix gets
# a usable subspace, and capped by the federation (`/status` `rank`). Both sides compute it from the base
# shapes, so a cold client knows the map before any global exists.
RANK_FLOOR = 8
RANK_DIVISOR = 128


def module_of(key: str) -> str | None:
    """The base module a factor key belongs to, or None if it is not a factor key at all."""
    for suffix in (LORA_A, LORA_B):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def phase_of(epoch: int) -> str:
    """Even epochs train B (against the frozen A), odd ones train A — derived, never stored."""
    return "B" if epoch % 2 == 0 else "A"


def rank_for(out_dim: int, in_dim: int, cap: int) -> int:
    """The federation's rank for ONE module: wider matrices earn more rank (a 4096² q_proj gets the cap,
    a GQA-shrunk v_proj the floor), never above the matrix's own full rank."""
    full = min(out_dim, in_dim)
    return min(max(full // RANK_DIVISOR, RANK_FLOOR), cap, full)


def rank_map(base: dict, cap: int) -> dict:
    """{module: rank} for a whole {module: (out, in)} shape map."""
    return {module: rank_for(out_dim, in_dim, cap) for module, (out_dim, in_dim) in base.items()}


def init_A(model_id: str, module: str, rank: int, in_dim: int) -> torch.Tensor:
    """The frozen A a fresh federation starts from, seeded from a SHAKE-256 XOF over
    (model_id, module, shape) so client and server derive the identical matrix. Uniform in ±1/√in,
    matching PEFT's kaiming-uniform lora_A init. The exact expansion is part of the contract."""
    raw = hashlib.shake_256(f"{model_id}|{module}|{rank}x{in_dim}".encode()).digest(rank * in_dim * 4)
    u = torch.frombuffer(bytearray(raw), dtype=torch.int32).to(torch.float64)
    u = (u + 2.0 ** 31) / 2.0 ** 32           # int32 residues -> [0, 1)
    bound = 1.0 / in_dim ** 0.5
    return ((2.0 * u - 1.0) * bound).to(torch.float32).reshape(rank, in_dim)


def base_to_json(shapes: dict) -> str:
    """The `base` upload stamp: every module's (out, in) verbatim. A one-factor payload pins only one
    dimension per module, and the server needs `in` to seed A and to derive the rank map."""
    return json.dumps([[m, list(s)] for m, s in sorted(shapes.items())])


def _dp_noise(A, B, z: float, generator):
    """Additive DP-bootstrap noise for B@A: B@N_A + N_B@A only — the linear terms of perturbing both
    factors, not the full (B+N_B)@(A+N_A) product. The dropped N_B@N_A cross term is a product of two
    independent Gaussians, which is itself not Gaussian (heavy-tailed, Bessel-K shaped) — keeping it
    would make the dense noise non-Gaussian and add variance for no protection benefit. B@N_A and N_B@A
    alone already keep both row(A) and col(B) unpredictable (row(N_A)/col(N_B) are random directions
    unrelated to A/B), which is the actual property "both factors noised" is for — so nothing about the
    original both-factor rationale is lost, and this is strictly less noisy than perturbing both factors
    before multiplying. Per-factor σ = z·rms(factor), so z is a relative multiplier rather than an
    absolute tied to a model's weight scale (data-dependent scale — not a formal sensitivity bound;
    still faux-DP, see delta.py module docstring)."""
    sigma_A = z * A.pow(2).mean().sqrt()
    sigma_B = z * B.pow(2).mean().sqrt()
    N_A = torch.randn(A.shape, generator=generator) * sigma_A
    N_B = torch.randn(B.shape, generator=generator) * sigma_B
    return B @ N_A + N_B @ A


def densify(delta: dict, *, noise_z: float = 0.0, generator=None) -> dict:
    """{module: dense ΔW = scaling·(B@A [+ noise])} from a round's PEFT factors (`delta["weights"]` =
    peft state dict, `["scaling"]` = alpha/r). noise_z>0 adds DP-bootstrap noise (see `_dp_noise`) —
    exactly Gaussian per dense entry, unlike naively multiplying two independently-noised factors."""
    sd, scaling = delta["weights"], float(delta["scaling"])
    out = {}
    for key, A in sd.items():
        if not key.endswith(".lora_A.weight"):
            continue
        mod = key[: -len(".lora_A.weight")]
        B   = sd[mod + ".lora_B.weight"]
        Af, Bf = A.float(), B.float()
        dense = Bf @ Af
        if noise_z:
            dense = dense + _dp_noise(Af, Bf, noise_z, generator)
        out[mod] = (scaling * dense).to(B.dtype)
    return out


def compat_from_shapes(shapes: dict) -> str:
    """The compat digest from an already-extracted {module: (out, in)} map. Split out so the server can
    recompute it while rebuilding the global per-module (it has shapes, not whole tensors)."""
    blob = ";".join(f"{m}:{s[0]}x{s[1]}" for m, s in sorted(shapes.items()))
    return hashlib.sha1(blob.encode()).hexdigest()


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
    return compat_from_shapes(shapes)


def _read_metadata(buf: bytes) -> dict:
    # safetensors layout: u64 LE header length, then the JSON header (whose "__metadata__" holds our
    # str→str fields). load() drops it, so parse the header directly rather than round-tripping a file.
    n = struct.unpack("<Q", buf[:8])[0]
    return json.loads(buf[8 : 8 + n]).get("__metadata__", {})


def to_bytes(tensors: dict, model_id: str, meta: dict | None = None) -> bytes:
    """Serialize a tensor dict (one factor's Δ for an upload, both factors for a broadcast) with
    model_id + the base-compat digest; `meta` carries the rest of the stamp (base, epoch, …). A
    one-factor payload pins only one dimension per module, so the digest comes from `base` when present."""
    meta = dict(meta or {})
    compat = (compat_from_shapes({m: tuple(s) for m, s in json.loads(meta["base"])}) if "base" in meta
              else compat_hash(tensors))
    return st_save(tensors, metadata={"model_id": model_id, "compat": compat, **meta})


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
