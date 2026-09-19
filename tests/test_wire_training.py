"""Tests for the wrapper's automatic training round: the factor-contract upload (federated/client.py
contribute_factor), captured-conversation discovery + consumption (runtime/train.py), and the REINFORCE++
round over captured chats (training/wire_trainer.py).
Run with:  PYTHONPATH=src python -m pytest tests/test_wire_training.py

CPU-only and download-free: a tiny Llama + a byte-level BPE tokenizer are built from scratch into tmp_path,
the PII detector and all federation I/O are monkeypatched.
"""
import json, os

import pytest
import torch

from roger.federated import client as fed_client, delta as delta_mod, transport
from roger.runtime import capture, train as rt_train


# --- contribute_factor ---------------------------------------------------------------------

_BASE = {"base_model.model.model.layers.0.self_attn.q_proj": (6, 4)}
_KEY = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"


def _update(scale=0.01):
    torch.manual_seed(0)
    return {_KEY: torch.randn(6, 2) * scale}


def test_contribute_factor_busy_stamps_and_drops_stale(monkeypatch):
    sent = []
    monkeypatch.setattr(transport, "register_and_peers", lambda url, pub, mid: ("rid", "tok", [pub], 3))
    monkeypatch.setattr(transport, "contribute", lambda url, blob: sent.append(blob) or "ok")
    assert fed_client.contribute_factor("http://x", _update(), _BASE, 3, "tiny") is True
    _, meta = delta_mod.from_bytes(sent[0])
    # the upload stamp the server's check_upload reads: base shapes, the epoch, the factor-key spec
    assert json.loads(meta["base"]) == [[m, list(s)] for m, s in _BASE.items()]
    assert meta["epoch"] == "3" and meta["compat"] == delta_mod.compat_from_shapes(_BASE)
    assert json.loads(meta["spec"]) == [[_KEY, [6, 2]]]
    # the cohort sealed in a later epoch: the update is dropped before upload, not voided after
    assert fed_client.contribute_factor("http://x", _update(), _BASE, 2, "tiny") is False
    assert len(sent) == 1
    monkeypatch.setattr(transport, "register_and_peers", lambda url, pub, mid: None)
    assert fed_client.contribute_factor("http://x", _update(), _BASE, 3, "tiny") is False


def test_contribute_factor_bootstrap_noises_and_clips(monkeypatch):
    sent = []
    monkeypatch.setattr(transport, "register_and_peers", lambda *a: pytest.fail("bootstrap needs no cohort"))
    monkeypatch.setattr(transport, "contribute_dp", lambda url, blob: sent.append(blob) or "ok")
    big = _update(scale=100.0)                                  # far above CLIP_NORM
    assert fed_client.contribute_factor("http://x", big, _BASE, 0, "tiny", mode="bootstrap") is True
    tensors, meta = delta_mod.from_bytes(sent[0])
    t = tensors[_KEY]
    assert meta["epoch"] == "0" and "base" in meta and torch.isfinite(t).all()
    assert not torch.allclose(t, big[_KEY])                     # noised
    # clipped to CLIP_NORM, then noised at σ = DP_Z·rms: well below the raw norm
    assert float(t.norm()) < 3 * fed_client.CLIP_NORM


# --- conversations on disk -----------------------------------------------------------------

def _write(d, name, msgs, reply, served="m.gguf", scores=None, signals=None):
    rec = {"endpoint": "/v1/chat/completions", "served_model": served, "request": {"messages": msgs},
           "response": {"choices": [{"message": {"role": "assistant", "content": reply}}]},
           "tool_signals": signals or {}}
    if scores is not None:
        rec["self_eval"] = {"scores": scores}
    path = os.path.join(d, name)
    with open(path, "w") as f:
        json.dump(rec, f)
    return path


@pytest.fixture
def msgs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "state_dir", lambda: str(tmp_path))
    d = capture.messages_dir("llama-server")
    os.makedirs(d)
    return d


def test_conversations_keep_the_newest_file_and_track_its_prefixes(msgs_dir):
    u1 = [{"role": "user", "content": "hi"}]
    first = _write(msgs_dir, "1.json", u1, "hello")                     # superseded by 2.json
    u2 = u1 + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]
    newest = _write(msgs_dir, "2.json", u2, "sure", scores={"accuracy": 1.0})
    other = _write(msgs_dir, "3.json", [{"role": "user", "content": "x"}], "y", scores={"accuracy": 0.0})
    _write(msgs_dir, "4.json", [{"role": "user", "content": "q"}], "a")   # ungraded: not ready
    _write(msgs_dir, "5.json", u1, "hello", served="other.gguf", scores={"accuracy": 1.0})
    convs = rt_train.conversations("llama-server", "m.gguf")
    by = {c["path"]: c for c in convs}
    assert set(by) == {newest, other, os.path.join(msgs_dir, "4.json")}
    assert by[newest]["superseded"] == [first]
    ready = rt_train.ready(convs)
    assert {c["path"] for c in ready} == {newest, other}
    rt_train.discard(ready)
    assert sorted(os.listdir(msgs_dir)) == ["4.json", "5.json"]


def _gate(monkeypatch, result, accepted=True, n=2):
    calls = {}
    monkeypatch.setattr(rt_train, "_federation",
                        lambda feds, served: (("http://x", "org/m", {"mode": "busy"}), []))
    monkeypatch.setattr(rt_train, "_refresh", lambda url, served, mid: None)
    monkeypatch.setattr(transport, "federation_status", lambda url, mid: {"mode": "busy", "epoch": 0,
                                                                          "phase": "B", "rank": 4})
    from roger.training import wire_trainer

    def fake_train(records, source, model_id, blob, status):
        calls["train"] = (len(records), source, model_id, status["phase"])
        if isinstance(result, BaseException):
            raise result
        return result
    monkeypatch.setattr(wire_trainer, "train", fake_train)
    monkeypatch.setattr(fed_client, "contribute_factor",
                        lambda *a: calls.setdefault("upload", a) and accepted)
    return calls


def _two_ready(msgs_dir):
    _write(msgs_dir, "1.json", [{"role": "user", "content": "a"}], "b", scores={"accuracy": 1.0})
    _write(msgs_dir, "2.json", [{"role": "user", "content": "c"}], "d", scores={"accuracy": -1.0})


_OK = {"trained": True, "update": {}, "base": {}, "epoch": 0, "phase": "B", "n_episodes": 2,
       "mean_return": 0.0}
_CFG = {"federations": ["http://x"], "contribute": True, "train_every": 2}


def test_maybe_train_waits_for_enough_conversations(msgs_dir, monkeypatch, capsys):
    calls = _gate(monkeypatch, _OK)
    _write(msgs_dir, "1.json", [{"role": "user", "content": "a"}], "b", scores={"accuracy": 1.0})
    rt_train.maybe_train("llama-server", "m.gguf", _CFG)
    assert "train" not in calls
    rt_train.maybe_train("llama-server", "m.gguf", dict(_CFG, contribute=False, train_every=1))
    assert "train" not in calls                                # leech: no gradient to send anywhere


def test_maybe_train_uploads_then_deletes(msgs_dir, monkeypatch, capsys):
    import sys
    calls = _gate(monkeypatch, _OK)
    _two_ready(msgs_dir)
    rt_train.maybe_train("llama-server", "m.gguf", _CFG, out=sys.stderr)   # capsys swaps sys.stderr
    assert calls["train"] == (2, "m.gguf", "org/m", "B") and "upload" in calls
    assert os.listdir(msgs_dir) == []
    assert "Ctrl-C to skip" in capsys.readouterr().err


@pytest.mark.parametrize("result,accepted", [(_OK, False), (KeyboardInterrupt(), True),
                                             (RuntimeError("boom"), True),
                                             ({"trained": False, "reason": "x"}, True)])
def test_maybe_train_keeps_data_unless_accepted(msgs_dir, monkeypatch, result, accepted):
    _gate(monkeypatch, result, accepted)
    _two_ready(msgs_dir)
    rt_train.maybe_train("llama-server", "m.gguf", _CFG)
    assert len(os.listdir(msgs_dir)) == 2


# --- the round over a tiny model --------------------------------------------------------------

_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
             "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(["hello there, how are you? fine thanks"] * 20, trainers.BpeTrainer(
        vocab_size=320, special_tokens=["<s>", "</s>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>")
    tok.chat_template = _TEMPLATE
    d = str(tmp_path / "tiny")
    tok.save_pretrained(d)
    torch.manual_seed(0)
    LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                 max_position_embeddings=256)).save_pretrained(d)
    from roger.training import privacy_filter
    monkeypatch.setattr(privacy_filter, "detect_pii_spans", lambda text: [])
    return d, tok


def _records(n=4):
    recs = []
    for i in range(n):
        msgs = [{"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "how are you?"},
                {"role": "user", "content": f"fine{i}"}]
        recs.append({"_path": f"{i}.json", "request": {"messages": msgs},
                     "response": {"choices": [{"message": {"role": "assistant", "content": "fine thanks"}}]},
                     "self_eval": {"scores": {"accuracy": (-1.0) ** i, "efficiency": 0.0}},
                     "tool_signals": {"1": -0.1} if i == 0 else {}})
    return recs


def test_episode_spans_are_the_assistant_turns(tiny_model):
    from roger.training import wire_trainer
    _, tok = tiny_model
    ep = wire_trainer.episode(tok, _records(1)[0])
    spans = [tok.decode(ep["seq"][e["gen_start"]: e["gen_start"] + len(e["masks"])]) for e in ep["traj"]]
    assert spans == ["how are you?<|im_end|>\n", "fine thanks<|im_end|>\n"]
    assert wire_trainer.reward(_records(1)[0]) == pytest.approx(0.5 - 0.1)
    rec = _records(1)[0]
    rec["self_eval"]["reactions"] = {"1": -1.0, "3": 0.0}                     # mean -0.5
    assert wire_trainer.reward(rec) == pytest.approx(0.5 - 0.5 - 0.1)


def test_train_updates_only_the_phase_factor_from_the_cold_start(tiny_model):
    from roger.training import wire_trainer
    d, _ = tiny_model
    res = wire_trainer.train(_records(), d, "org/tiny", None, {"epoch": 0, "phase": "B", "rank": 4},
                             lr=1e-2)
    assert res["trained"] and res["epoch"] == 0 and res["n_episodes"] == 4
    keys = set(res["update"])
    assert keys and all(k.endswith(delta_mod.LORA_B) for k in keys)
    # canonical decoder-path keys, per-module rank from the shared rule, a real (nonzero) step
    q = "base_model.model.model.layers.0.self_attn.q_proj"
    assert res["base"][q] == (32, 32) and res["base"][q.replace("q_proj", "v_proj")] == (16, 32)
    assert tuple(res["update"][q + delta_mod.LORA_B].shape) == (32, 4)
    assert any(float(t.abs().sum()) > 0 for t in res["update"].values())


def test_train_starts_from_the_global_and_trains_A_in_an_A_epoch(tiny_model, monkeypatch):
    from roger.training import wire_trainer
    d, _ = tiny_model
    torch.manual_seed(1)
    glob, base = {}, {}
    for layer in range(2):
        for name, out_dim in (("q_proj", 32), ("v_proj", 16)):
            m = f"base_model.model.model.layers.{layer}.self_attn.{name}"
            glob[m + delta_mod.LORA_A] = torch.randn(4, 32) * 0.1
            glob[m + delta_mod.LORA_B] = torch.randn(out_dim, 4) * 0.1
            base[m] = (out_dim, 32)
    blob = delta_mod.to_bytes(glob, "org/tiny", {"scaling": "1", "epoch": "1"})
    seen = {}
    real = wire_trainer.trainer.reinforce

    def spy(model, eps, adv, opt, trainable, **kw):
        # at the start of the round the adapter IS the global
        lora = model.get_submodule("base_model.model.model.layers.0.self_attn.q_proj")
        seen["A"] = lora.lora_A["default"].weight.detach().float().cpu()
        seen["B_grad"] = lora.lora_B["default"].weight.requires_grad
        return real(model, eps, adv, opt, trainable, **kw)
    monkeypatch.setattr(wire_trainer.trainer, "reinforce", spy)
    res = wire_trainer.train(_records(), d, "org/tiny", blob, {"epoch": 1, "phase": "A", "rank": 4}, lr=1e-2)
    q = "base_model.model.model.layers.0.self_attn.q_proj"
    assert torch.allclose(seen["A"], glob[q + delta_mod.LORA_A]) and seen["B_grad"] is False
    assert all(k.endswith(delta_mod.LORA_A) for k in res["update"]) and res["base"] == base


def test_train_refuses_a_global_for_another_shape(tiny_model):
    from roger.training import wire_trainer
    d, _ = tiny_model
    m = "base_model.model.model.layers.0.self_attn.q_proj"
    blob = delta_mod.to_bytes({m + delta_mod.LORA_A: torch.zeros(4, 99), m + delta_mod.LORA_B: torch.zeros(32, 4)},
                              "org/tiny")
    res = wire_trainer.train(_records(), d, "org/tiny", blob, {"epoch": 0, "phase": "B", "rank": 4})
    assert res["trained"] is False and "doesn't fit" in res["reason"]


def test_federation_follows_config_order_and_reports_skips(monkeypatch):
    statuses = {"http://down": {}, "http://other": {"mode": "bootstrap", "models": ["org/else"]},
                "http://old": {"mode": "busy", "models": None, "min_client": 10 ** 6},
                "http://ok": {"mode": "busy", "models": ["org/m"]}}
    monkeypatch.setattr(transport, "federation_status", lambda url, mid: statuses[url])
    fed, skipped = rt_train._federation(["http://down", "http://other", "http://old", "http://ok"], "x/m.gguf")
    assert fed[:2] == ("http://ok", "org/m")
    assert [u for u, _ in skipped] == ["http://down", "http://other", "http://old"]
    assert "unreachable" in skipped[0][1] and "doesn't train m.gguf" in skipped[1][1] and "newer" in skipped[2][1]


def test_maybe_train_names_the_skipped_preferred_federation(msgs_dir, monkeypatch):
    import io
    _gate(monkeypatch, _OK)
    monkeypatch.setattr(rt_train, "_federation", lambda feds, served: (
        ("http://b", "org/m", {"mode": "busy"}), [("http://a", "unreachable")]))
    _two_ready(msgs_dir)
    out = io.StringIO()
    rt_train.maybe_train("llama-server", "m.gguf", _CFG, out=out)
    assert "not training for http://a: unreachable." in out.getvalue()
    assert "training for http://b instead" in out.getvalue()
