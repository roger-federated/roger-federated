"""Tests for federated gradient sharing (delta densify, secure-aggregation masking, transport).
Run with:  PYTHONPATH=src python -m pytest tests/test_federated.py

All CPU-only and download-free: a tiny Llama from config exercises the adapter / fold paths, and
transport is monkeypatched so no network is touched.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from roger.federated import delta as delta_mod, secure_agg, client as fed_client, transport


# --- ΔW densification + compatibility --------------------------------------------------------

def _fake_delta(out=6, in_=4, r=2, scaling=2.0, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(r, in_); B = torch.randn(out, r)   # B≠0 here (post-step); fresh init would be 0
    return {"weights": {"m.lora_A.weight": A, "m.lora_B.weight": B},
            "scaling": scaling, "model_id": "tiny"}


def test_densify_matches_factors():
    d = _fake_delta()
    A, B, s = d["weights"]["m.lora_A.weight"], d["weights"]["m.lora_B.weight"], d["scaling"]
    dense = delta_mod.densify(d)
    assert torch.allclose(dense["m"], s * (B @ A), atol=1e-5)
    print("PASS test_densify_matches_factors")


def test_compat_hash_factors_eq_dense_and_detects_shape():
    d = _fake_delta(out=6, in_=4)
    factors_hash = delta_mod.compat_hash(d["weights"])
    dense_hash   = delta_mod.compat_hash(delta_mod.densify(d))
    assert factors_hash == dense_hash                       # same base ⇒ same hash either form
    assert delta_mod.compat_hash(_fake_delta(out=8, in_=4)["weights"]) != factors_hash
    print("PASS test_compat_hash_factors_eq_dense_and_detects_shape")


def test_bytes_roundtrip():
    dense = delta_mod.densify(_fake_delta())
    buf = delta_mod.to_bytes(dense, "tiny")
    got, meta = delta_mod.from_bytes(buf)
    assert torch.allclose(got["m"], dense["m"])
    assert meta["model_id"] == "tiny" and meta["compat"] == delta_mod.compat_hash(dense)
    print("PASS test_bytes_roundtrip")


# --- secure aggregation ----------------------------------------------------------------------

def test_quantize_dequantize_roundtrip():
    dense = {"m": torch.randn(6, 4) * 0.1}
    q, spec = secure_agg.quantize(dense)
    back = secure_agg.dequantize(q, spec)
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
    recovered = secure_agg.dequantize(agg, spec)["m"]
    expected = sum(p["m"] for p in payloads)
    assert torch.allclose(recovered, expected, atol=1e-2), (recovered, expected)
    print("PASS test_mask_cancellation")


def test_mask_noop_when_alone():
    # A single participant has no peer to cancel against ⇒ mask is a no-op (degenerate round).
    (priv, pub) = secure_agg.gen_keypair()
    q, _ = secure_agg.quantize({"m": torch.randn(4, 4) * 0.1})
    assert torch.equal(secure_agg.mask(q, priv, [pub]), q)
    print("PASS test_mask_noop_when_alone")


# --- fold the dense global into the base weights ---------------------------------------------

def test_fold_into_adds_and_skips_mismatch():
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    tgt   = "model.layers.0.self_attn.q_proj"
    other = "model.layers.1.mlp.gate_proj"
    w0     = model.get_submodule(tgt).weight.detach().clone()
    other0 = model.get_submodule(other).weight.detach().clone()
    dW = torch.randn_like(w0)
    deltas = {f"base_model.model.{tgt}": dW,                       # PEFT-prefixed key (tests _base_key)
              "base_model.model.model.layers.0.self_attn.v_proj": torch.randn(5, 5)}  # wrong shape → skip
    n = delta_mod.fold_into(model, deltas)
    assert n == 1                                                  # only the valid one folded
    assert torch.allclose(model.get_submodule(tgt).weight.float(), (w0 + dW).float(), atol=1e-2)
    assert torch.equal(model.get_submodule(other).weight, other0)  # untouched
    print("PASS test_fold_into_adds_and_skips_mismatch")


# --- client gating + orchestration (transport monkeypatched) ---------------------------------

def test_is_leeching_and_should_train():
    assert fed_client.is_leeching({"federations": ["u"], "contribute": False})
    assert not fed_client.is_leeching({"federations": [], "contribute": False})
    assert fed_client.should_train({"federations": ["u"], "contribute": True})
    assert not fed_client.should_train({"federations": ["u"], "contribute": False})
    assert not fed_client.should_train({"federations": [], "contribute": True})
    print("PASS test_is_leeching_and_should_train")


def test_contribute_delta_uploads(monkeypatch):
    sent = []
    monkeypatch.setattr(transport, "register_and_peers", lambda url, pub, mid: ("rid", [pub]))
    monkeypatch.setattr(transport, "contribute", lambda url, blob: sent.append((url, blob)) or "ok")
    fed_client.contribute_delta(_fake_delta(), {"federations": ["http://x"], "contribute": True})
    assert len(sent) == 1 and sent[0][0] == "http://x"
    # leech / no-federation are no-ops
    fed_client.contribute_delta(_fake_delta(), {"federations": ["http://x"], "contribute": False})
    fed_client.contribute_delta(_fake_delta(), {"federations": [], "contribute": True})
    assert len(sent) == 1
    print("PASS test_contribute_delta_uploads")


def test_maybe_daily_pull_persists_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, "_state_path", lambda url: str(tmp_path / "fed.json"))
    saved = []
    monkeypatch.setattr(transport, "pull", lambda url, cur, mid: (b"blob", "c1"))
    monkeypatch.setattr(transport, "save_global", lambda url, blob: saved.append(blob))
    cfg = {"federations": ["http://x"], "model_id": "tiny"}
    assert fed_client.maybe_daily_pull(cfg) is True       # fetched + persisted
    assert saved == [b"blob"]
    assert fed_client.maybe_daily_pull(cfg) is False       # same UTC day ⇒ no re-pull
    assert saved == [b"blob"]
    print("PASS test_maybe_daily_pull_persists_blob")


def test_pending_globals_sums_federations(monkeypatch):
    d1 = delta_mod.densify(_fake_delta(seed=1))
    d2 = delta_mod.densify(_fake_delta(seed=2))
    blobs = {"http://a": delta_mod.to_bytes(d1, "tiny"), "http://b": delta_mod.to_bytes(d2, "tiny")}
    monkeypatch.setattr(transport, "load_global", lambda url: blobs.get(url))
    out = fed_client.pending_globals({"federations": ["http://a", "http://b"]})
    assert torch.allclose(out["m"], d1["m"] + d2["m"], atol=1e-5)
    monkeypatch.setattr(transport, "load_global", lambda url: None)
    assert fed_client.pending_globals({"federations": ["http://a"]}) is None   # none persisted
    assert fed_client.pending_globals({"federations": []}) is None             # no federation
    print("PASS test_pending_globals_sums_federations")


# --- trainer adapter: single fresh adapter, ΔW extractable -----------------------------------

def test_attach_single_adapter_and_extract():
    """One fresh LoRA adapter (no global/inference adapter); B=0 ⇒ ΔW starts at 0; a step makes it
    non-trivial. The folded global lives in the base weights, not in an adapter."""
    from roger.training import lora_utils
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = lora_utils.attach_lora(LlamaForCausalLM(cfg), targets="all-linear")
    assert len(model.peft_config) == 1                                  # exactly one adapter
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)           # only LoRA factors train

    sd0 = lora_utils.local_state_dict(model)
    d0 = delta_mod.densify({"weights": sd0, "scaling": 2.0, "model_id": "tiny"})
    assert all(float(v.abs().sum()) == 0.0 for v in d0.values())        # B=0 ⇒ ΔW=0

    model.train()
    model(input_ids=torch.randint(0, 64, (1, 8)), labels=torch.randint(0, 64, (1, 8))).loss.backward()
    for p in (p for p in model.parameters() if p.requires_grad):
        if p.grad is not None:
            p.data -= 0.1 * p.grad
    d1 = delta_mod.densify({"weights": lora_utils.local_state_dict(model), "scaling": 2.0, "model_id": "tiny"})
    assert any(float(v.abs().sum()) > 0.0 for v in d1.values())
    print("PASS test_attach_single_adapter_and_extract")


if __name__ == "__main__":
    test_densify_matches_factors()
    test_compat_hash_factors_eq_dense_and_detects_shape()
    test_bytes_roundtrip()
    test_quantize_dequantize_roundtrip()
    test_mask_cancellation()
    test_mask_noop_when_alone()
    test_fold_into_adds_and_skips_mismatch()
    test_is_leeching_and_should_train()
