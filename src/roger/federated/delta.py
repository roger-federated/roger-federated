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


def compat_from_shapes(shapes: dict) -> str:
    """The compat digest from an already-extracted {module: (out, in)} map. Split out so the server can
    recompute it while rebuilding the global per-module (it has shapes, not whole tensors)."""
    blob = ";".join(f"{m}:{s[0]}x{s[1]}" for m, s in sorted(shapes.items()))
    return hashlib.sha1(blob.encode()).hexdigest()


def compat_hash(tensors: dict) -> str:
    """Stable digest of the base architecture this update targets: sorted module → (out, in), read off
    the factors (out from lora_B's rows, in from lora_A's columns). A one-factor upload only pins one
    of the two dimensions, which is why `base` travels alongside; this stays the canonical digest of a
    complete factor pair (a stored global, a broadcast)."""
    shapes = {}
    for key, t in tensors.items():
        if key.endswith(LORA_A):
            shapes.setdefault(key[: -len(LORA_A)], [None, None])[1] = t.shape[1]    # in
        elif key.endswith(LORA_B):
            shapes.setdefault(key[: -len(LORA_B)], [None, None])[0] = t.shape[0]    # out
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
