"""secure_agg.py — the client's half of Bonawitz et al. 2017 secure aggregation
(ACM CCS, doi:10.1145/3133956.3133982), so the server learns only Σ ΔW, never any individual ΔW.

Each user i uploads, in fixed-point integers mod R:

    g_i = q(ΔW_i) + Σ_k ε_{ik}·(2·1_{i>k} − 1)   (mod R)

where ε_{ik} = ε_{ki} is a *pairwise* noise vector both parties derive locally from their shared
X25519 (EC-DH) secret — never transmitted — expanded to payload length by a SHAKE-256 PRG. The sign
term (+1 for the larger participant id, −1 for the smaller, ordered by raw public key) is
antisymmetric, so when the server sums every g_i each pair's masks cancel exactly and Σ g_i = Σ q(ΔW_i).
Working mod R is what makes a single g_i uniform over Z_R (the privacy guarantee); the cancellation
itself is exact regardless. Quantize→mask here on the client; the server dequantizes after summing.

The peer public keys are distributed by the federation server (the user wants keys collected
centrally, not peer-to-peer); the self-mask b_u term and Shamir-shared dropout recovery from the
full protocol are server-coordinated and live with the server (see federated_server_requirements).
"""
import hashlib

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

# Fixed-point: ΔW is clipped to O(1), so 16 fractional bits is ample precision; R = 2^32 leaves a
# huge margin before the (cancelled) masks could wrap a real aggregate. Both must match the server.
SCALE = 1 << 16
R     = 1 << 32


def gen_keypair() -> tuple[X25519PrivateKey, bytes]:
    priv = X25519PrivateKey.generate()
    return priv, priv.public_key().public_bytes_raw()


def quantize(tensors: dict) -> tuple[torch.Tensor, list]:
    """Flatten {module: ΔW} (sorted-key order, so every client lays out the vector identically) to a
    1-D int64 residue vector mod R. Returns (flat, spec) where spec rebuilds the dict server-side."""
    spec, parts = [], []
    for key in sorted(tensors):
        t = tensors[key]
        spec.append((key, tuple(t.shape)))
        parts.append(t.reshape(-1).float())
    flat = torch.cat(parts) if parts else torch.zeros(0)
    q = torch.round(flat * SCALE).to(torch.int64) % R
    return q, spec


# `dequantize` (the inverse of quantize) is server-only: the aggregation server sums the masked
# uploads then maps the residues back to signed reals. It lives in the separate roger-server repo
# (roger_server/secure_agg.py), kept in lockstep with SCALE/R/quantize/mask here.


def _prg(secret: bytes, length: int) -> torch.Tensor:
    """SHAKE-256 XOF over the shared secret → `length` int64 residues mod R. Deterministic, so both
    parties to a pair produce the identical ε."""
    raw = hashlib.shake_256(secret).digest(length * 8)
    arr = torch.frombuffer(bytearray(raw), dtype=torch.int64)
    return arr % R


def mask(flat: torch.Tensor, my_priv: X25519PrivateKey, peer_pubs: list[bytes]) -> torch.Tensor:
    """Add the antisymmetric pairwise masks for this round's peers. `peer_pubs` may include our own
    key (skipped). Cancels against every other participant when the server sums all uploads."""
    my_pub = my_priv.public_key().public_bytes_raw()
    out, L = flat.clone(), flat.numel()
    for pub in peer_pubs:
        if pub == my_pub:
            continue
        eps  = _prg(my_priv.exchange(X25519PublicKey.from_public_bytes(pub)), L)
        sign = 1 if my_pub > pub else -1          # +1 for the larger id, −1 for the smaller
        out  = (out + sign * eps) % R
    return out % R
