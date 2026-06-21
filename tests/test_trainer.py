"""Tests for the LoRA REINFORCE++ trainer's log-prob machinery.
Run with:  PYTHONPATH=src python -m pytest tests/test_trainer.py

These guard the two subtle parts: the allowed-set masking and the off-by-one in the
teacher-forced forward (token at absolute index k is predicted by logits[k-1]). A tiny model is
built from a config so the test needs no model download and runs on CPU.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from roger.training.trainer import _apply_masks, _new_logps
from roger.agency.rollout_utils import _old_logps


def test_old_logps_matches_manual():
    # Recording-side behaviour log-prob: constrained log-softmax over each step's logits, gather
    # the sampled token. This is the value the trainer's importance ratio trusts as `old`.
    torch.manual_seed(1)
    V = 32
    step_logits = [torch.randn(1, V) for _ in range(3)]
    tokens = torch.tensor([5, 9, 2])
    masks = [None, [9, 1, 4], [2, 7]]   # constrained sets must contain the sampled token
    got = _old_logps(step_logits, masks, tokens, torch.device("cpu"))
    for t in range(3):
        row = step_logits[t].squeeze(0).clone()
        if masks[t] is not None:
            neg = torch.full((V,), float("-inf")); neg[masks[t]] = 0.0; row = row + neg
        assert torch.allclose(got[t], torch.log_softmax(row, -1)[int(tokens[t])], atol=1e-6)
    assert not got.requires_grad and got.device.type == "cpu"
    print("PASS test_old_logps_matches_manual")


def test_apply_masks():
    logits = torch.randn(3, 8)
    masks = [None, [0, 2], None]
    out = _apply_masks(logits.clone(), masks)
    # None rows pass through untouched; constrained rows are -inf off the allowed set, unchanged on it.
    assert torch.equal(out[0], logits[0])
    assert torch.equal(out[2], logits[2])
    allowed = {0, 2}
    for j in range(8):
        if j in allowed:
            assert out[1, j] == logits[1, j]
        else:
            assert out[1, j] == float("-inf")
    print("PASS test_apply_masks")


def test_new_logps_alignment():
    torch.manual_seed(0)
    V = 64
    cfg = LlamaConfig(vocab_size=V, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(cfg).eval()
    device = torch.device("cpu")

    T, g0, n = 12, 5, 4
    seq = torch.randint(0, V, (T,))
    gen_token_ids = seq[g0:g0 + n].clone()
    # Mix unconstrained and constrained steps; constrained sets must contain the sampled token.
    masks = [None,
             [int(gen_token_ids[1]), 3, 7],
             None,
             [int(gen_token_ids[3]), 1]]

    # Expected new log-probs computed independently with explicit pos-1 indexing, so a regression
    # to logits[g0:...] (dropping the off-by-one) would change these values and fail the test.
    with torch.no_grad():
        full = model(seq.unsqueeze(0)).logits[0].float()
    expected = []
    for t in range(n):
        row = full[g0 + t - 1].clone()
        if masks[t] is not None:
            neg = torch.full((V,), float("-inf"))
            neg[masks[t]] = 0.0
            row = row + neg
        expected.append(torch.log_softmax(row, -1)[int(gen_token_ids[t])])
    expected = torch.stack(expected)

    # No gen_token_ids in the entry: the trainer recovers the tokens as seq[gen_start:gen_start+n].
    ep = {"dir": "synthetic", "seq": seq,
          "traj": [{"gen_start": g0, "old_logp": expected.clone(), "masks": masks}]}
    new_lp, old_lp = _new_logps(model, ep, device)
    assert torch.allclose(new_lp, expected, atol=1e-5), (new_lp, expected)
    # old_lp is the stored tensor passed through (detached); equal to new here, so ratio == 1
    # when the training adapter matches the collector (the on-policy starting point).
    assert torch.allclose(old_lp, expected, atol=1e-5)
    assert not old_lp.requires_grad      # detached behaviour log-prob
    assert new_lp.requires_grad          # differentiable: gradient flows back to the policy
    print("PASS test_new_logps_alignment")


if __name__ == "__main__":
    test_apply_masks()
    test_old_logps_matches_manual()
    test_new_logps_alignment()
