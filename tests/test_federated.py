"""Tests for the federated wire contract: the factor layout + serialization (delta.py) and the
secure-aggregation crypto (secure_agg.py), plus the shared LoRA basis.
Run with:  PYTHONPATH=src python -m pytest tests/test_federated.py

All CPU-only and download-free. Both modules are mirrored BY HAND in the roger-server repo, so the
assertions here duplicate the ones in its tests/test_server.py on purpose: a hand-edit to the rank
rule, `init_A`, the flatten layout or the quantization must break a test on both sides.
The upload itself (client.contribute_factor) is covered by tests/test_wire_training.py.
"""
import json

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from roger.federated import delta as delta_mod, secure_agg


def _dequantize(flat, spec):
    """Local mirror of the inverse of secure_agg.quantize. The client package doesn't ship
    `dequantize` (it is server-only, in the roger-server repo), but the round-trip and mask-cancellation
    tests below still need it to check the quantize/mask math. Residues ≥ R/2 represent negatives."""
    R, SCALE = secure_agg.R, secure_agg.SCALE
    signed = flat.clone()
    signed[signed >= R // 2] -= R
    real, out, off = signed.float() / SCALE, {}, 0
    for key, shape in spec:
        n = 1
        for d in shape:
            n *= d
        out[key] = real[off : off + n].reshape(shape)
        off += n
    return out


# --- the factor contract ---------------------------------------------------------------------

_MOD = "base_model.model.model.layers.0.self_attn.q_proj"


def _factors(out=6, in_=4, r=2, seed=0):
    torch.manual_seed(seed)
    return {_MOD + delta_mod.LORA_A: torch.randn(r, in_),
            _MOD + delta_mod.LORA_B: torch.randn(out, r)}


def test_phase_alternates_and_epoch_zero_trains_b():
    # LoRA starts at B=0, so A's gradient would be zero in epoch 0: B must go first.
    assert delta_mod.phase_of(0) == "B" and delta_mod.phase_of(1) == "A"
    assert delta_mod.phase_of(4) == "B" and delta_mod.phase_of(7) == "A"
    assert delta_mod.SUFFIX["B"] == delta_mod.LORA_B and delta_mod.SUFFIX["A"] == delta_mod.LORA_A
    assert delta_mod.module_of(_MOD + delta_mod.LORA_B) == _MOD
    assert delta_mod.module_of(_MOD + ".weight") is None
    print("PASS test_phase_alternates_and_epoch_zero_trains_b")


def test_rank_rule_matches_the_server():
    # Same assertions as roger-server's tests/test_server.py: the map is a pure function of the base
    # shapes, so every member derives it identically before anyone trains.
    assert delta_mod.rank_for(4096, 4096, 16) == 16 and delta_mod.rank_for(1024, 4096, 16) == 8
    assert delta_mod.rank_for(4096, 4096, 64) == 32                    # cap not binding
    assert delta_mod.rank_for(2, 2, 16) == 2                           # never exceed the matrix itself
    assert delta_mod.rank_map({"q": (4096, 4096), "v": (1024, 4096)}, 16) == {"q": 16, "v": 8}
    print("PASS test_rank_rule_matches_the_server")


def test_init_a_is_derived_and_seed_dependent():
    # The frozen A of a cold federation is never transmitted: client and server must derive the same
    # matrix from (model_id, module, shape), and a different model must not reuse it.
    a1 = delta_mod.init_A("m", _MOD, 4, 32)
    assert tuple(a1.shape) == (4, 32) and a1.dtype is torch.float32
    assert torch.equal(a1, delta_mod.init_A("m", _MOD, 4, 32))
    assert not torch.equal(a1, delta_mod.init_A("other", _MOD, 4, 32))
    assert float(a1.abs().max()) <= 1.0 / 32 ** 0.5                    # PEFT's kaiming-uniform bound
    print("PASS test_init_a_is_derived_and_seed_dependent")


def test_compat_hash_reads_both_dims_off_the_factor_pair():
    f = _factors(out=6, in_=4)
    assert delta_mod.compat_hash(f) == delta_mod.compat_from_shapes({_MOD: (6, 4)})
    assert delta_mod.compat_hash(_factors(out=8, in_=4)) != delta_mod.compat_hash(f)
    print("PASS test_compat_hash_reads_both_dims_off_the_factor_pair")


def test_bytes_roundtrip_and_base_stamped_digest():
    f = _factors()
    buf = delta_mod.to_bytes(f, "tiny")
    got, meta = delta_mod.from_bytes(buf)
    assert torch.allclose(got[_MOD + delta_mod.LORA_A], f[_MOD + delta_mod.LORA_A])
    assert meta["model_id"] == "tiny" and meta["compat"] == delta_mod.compat_hash(f)
    # A one-factor upload pins only one dimension, so the digest must come from the `base` stamp.
    base = {_MOD: (6, 4)}
    one = {_MOD + delta_mod.LORA_B: f[_MOD + delta_mod.LORA_B]}
    stamp = {"base": delta_mod.base_to_json(base), "epoch": "3"}
    _, meta = delta_mod.from_bytes(delta_mod.to_bytes(one, "tiny", stamp))
    assert meta["compat"] == delta_mod.compat_from_shapes(base) == delta_mod.compat_hash(f)
    assert json.loads(meta["base"]) == [[_MOD, [6, 4]]] and meta["epoch"] == "3"
    print("PASS test_bytes_roundtrip_and_base_stamped_digest")


# --- secure aggregation ----------------------------------------------------------------------

def test_quantize_dequantize_roundtrip():
    dense = {"m": torch.randn(6, 4) * 0.1}
    q, spec = secure_agg.quantize(dense)
    back = _dequantize(q, spec)
    assert torch.allclose(back["m"], dense["m"], atol=1e-3)
    print("PASS test_quantize_dequantize_roundtrip")


def test_mask_cancellation():
    """The crux: Σ of masked uploads == Σ of raw payloads, while each individual upload is hidden."""
    N = 4
    torch.manual_seed(1)
    payloads = [{"m": torch.randn(6, 4) * 0.05} for _ in range(N)]
    keys = [secure_agg.gen_keypair() for _ in range(N)]
    pubs = [pub for _, pub in keys]

    masked, raw_q = [], []
    spec = None
    for (priv, _), p in zip(keys, payloads):
        q, spec = secure_agg.quantize(p)
        raw_q.append(q)
        masked.append(secure_agg.mask(q, priv, pubs))

    # Each masked upload differs from its raw quantization (it's been hidden).
    for q, mk in zip(raw_q, masked):
        assert not torch.equal(q % secure_agg.R, mk)
    # But the masks cancel in the sum, recovering the true aggregate.
    agg = (sum(masked) % secure_agg.R)
    recovered = _dequantize(agg, spec)["m"]
    expected = sum(p["m"] for p in payloads)
    assert torch.allclose(recovered, expected, atol=1e-2), (recovered, expected)
    print("PASS test_mask_cancellation")


def test_mask_noop_when_alone():
    # A single participant has no peer to cancel against ⇒ mask is a no-op (degenerate round).
    (priv, pub) = secure_agg.gen_keypair()
    q, _ = secure_agg.quantize({"m": torch.randn(4, 4) * 0.1})
    assert torch.equal(secure_agg.mask(q, priv, [pub]), q)
    print("PASS test_mask_noop_when_alone")


def test_flatten_layout_is_sorted_by_key():
    # The server locates each factor's slice by laying the keys out in sorted() order; a client that
    # flattened in dict order would desync every offset in the cohort.
    tensors = {"z": torch.zeros(2, 2), "a": torch.ones(3, 1)}
    _, spec = secure_agg.quantize(tensors)
    assert [k for k, _ in spec] == ["a", "z"]
    print("PASS test_flatten_layout_is_sorted_by_key")


# --- the shared LoRA basis -------------------------------------------------------------------

def test_federation_basis_is_q_v():
    """The federation basis is fixed to q/v (the secure-agg layout every member must share). It is NOT
    all-linear: the server stages + sums each factor, so a broad target set blows up per-round I/O."""
    from roger.training import lora_utils
    assert lora_utils.FED_TARGETS == ["q_proj", "v_proj"]
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = lora_utils.attach_lora(LlamaForCausalLM(cfg), targets=lora_utils.FED_TARGETS)
    assert len(model.peft_config) == 1                                  # exactly one adapter
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)           # only LoRA factors train
    wrapped = {n.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1] for n in trainable}
    assert wrapped == {"q_proj", "v_proj"}     # only q/v, never k/o/mlp/lm_head
    print("PASS test_federation_basis_is_q_v")


if __name__ == "__main__":
    test_phase_alternates_and_epoch_zero_trains_b()
    test_rank_rule_matches_the_server()
    test_init_a_is_derived_and_seed_dependent()
    test_compat_hash_reads_both_dims_off_the_factor_pair()
    test_bytes_roundtrip_and_base_stamped_digest()
    test_quantize_dequantize_roundtrip()
    test_mask_cancellation()
    test_mask_noop_when_alone()
    test_flatten_layout_is_sorted_by_key()
    test_federation_basis_is_q_v()
