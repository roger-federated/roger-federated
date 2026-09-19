"""trainer.py — the LoRA REINFORCE++ update itself.

The algorithm half of a training round, independent of where the episodes came from: teacher-force
the policy over each episode to recompute differentiable per-token log-probs, then apply a flat
REINFORCE++ step — episode return over all its tokens, batch-mean baseline, z-normed advantages,
PPO-clipped ratio vs the behaviour log-probs, no KL. PII is rewritten to surrogates first, so no
gradient ever sees it. The round that builds the episodes and does something with the resulting
update is training/wire_trainer.py.
"""
from contextlib import nullcontext

import torch

from roger.training import privacy_filter


def _apply_masks(logits: torch.Tensor, masks: list) -> torch.Tensor:
    """Re-impose each step's allowed-set on a [n, vocab] slice (None = full vocab). Out-of-place
    so the additive -inf never trips autograd's in-place checks."""
    add = torch.zeros_like(logits)
    for t, allowed in enumerate(masks):
        if allowed is None:
            continue
        row = torch.full((logits.size(-1),), float("-inf"), device=logits.device, dtype=logits.dtype)
        row[allowed] = 0.0
        add[t] = row
    return logits + add


def _new_logps(model, ep, device) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward -> (new_logp differentiable, old_logp detached), 1-D over every generated token.

    `logits_to_keep` (an index tensor) runs the head only at the generated positions, avoiding the
    [T, vocab] logits (~10 GB for a long session). Token k is predicted by hidden state k-1."""
    seq = ep["seq"].to(device)
    # positions k-1 for every generated token k, across all turns (only generated tokens are scored)
    keep = []
    for e in ep["traj"]:
        g0, n = int(e["gen_start"]), len(e["masks"])
        keep.extend(range(g0 - 1, g0 - 1 + n))
    keep = torch.tensor(keep, device=device)
    autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda" else nullcontext())
    with autocast:
        out = model(input_ids=seq.unsqueeze(0), attention_mask=torch.ones_like(seq).unsqueeze(0),
                    use_cache=False, logits_to_keep=keep)
    logits = out.logits[0].float()                          # [len(keep), vocab], in keep order
    new_parts, old_parts, off = [], [], 0
    for e in ep["traj"]:                                    # same order as keep; walk with `off`
        g0, n = int(e["gen_start"]), len(e["masks"])
        assert n == e["old_logp"].numel(), f"misaligned step in {ep['dir']}"
        turn_logits = _apply_masks(logits[off: off + n], e["masks"])
        toks = seq[g0: g0 + n].long()
        new_parts.append(torch.log_softmax(turn_logits, dim=-1).gather(-1, toks[:, None]).squeeze(-1))
        old_parts.append(e["old_logp"].to(device))
        off += n
    return torch.cat(new_parts), torch.cat(old_parts).detach()


def advantages(returns: torch.Tensor) -> torch.Tensor | None:
    """REINFORCE++ advantages: batch-mean baseline, z-normed; one scalar per episode. None when the
    returns have zero variance (nothing to prefer, so nothing to learn)."""
    centered = returns - returns.mean()
    if float(centered.std()) == 0.0:
        return None
    return centered / centered.std().clamp_min(1e-6)


def anonymize(eps: list, tokenizer) -> None:
    """Rewrite PII to surrogates in each episode's `seq` before any gradient sees it, and set
    `keep` (per generated token, in _new_logps order; 0 = a rewritten PII token). Frees the filter
    afterwards so it doesn't hold VRAM. Detect/load failures propagate rather than train on raw PII."""
    for ep in eps:
        new_seq, pii_pos = privacy_filter.anonymize_sequence(ep["seq"], tokenizer)
        ep["seq"] = new_seq
        keep = [0.0 if (int(e["gen_start"]) + j) in pii_pos else 1.0
                for e in ep["traj"] for j in range(len(e["masks"]))]
        ep["keep"] = torch.tensor(keep)
    privacy_filter.free_filter()


def reinforce(model, eps: list, adv: torch.Tensor, opt, trainable: list, *, epochs: int = 1,
              clip_eps: float = 0.2, max_grad_norm: float = 1.0) -> float:
    """The flat REINFORCE++ update over `eps` (each with `keep` set by `anonymize`): PPO-clipped ratio
    vs the stored behaviour log-probs, the episode's advantage broadcast over all its kept generated
    tokens, no KL. Returns the summed loss."""
    device = next(model.parameters()).device
    # 1/total_tokens scaling + per-episode backward() accumulation = exact token-mean, no padding.
    # total_tokens counts only kept tokens, so dropped PII positions don't skew the token-mean.
    total_tokens = int(sum(float(ep["keep"].sum()) for ep in eps))
    last_loss = 0.0
    for _ in range(max(1, epochs)):
        opt.zero_grad(set_to_none=True)
        for i, ep in enumerate(eps):
            new_lp, old_lp = _new_logps(model, ep, device)
            # Drop rewritten PII tokens before the ratio; zeroing post-hoc risks 0*inf=nan grads.
            keep      = ep["keep"].to(device).bool()
            new_lp, old_lp = new_lp[keep], old_lp[keep]
            ratio     = torch.exp(new_lp - old_lp)            # ~1 on fresh data; clip bites at epochs>1
            a         = adv[i].to(device)
            unclipped = ratio * a
            clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a
            loss      = -torch.sum(torch.min(unclipped, clipped)) / total_tokens   # token-mean via accumulation
            loss.backward()
            last_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        opt.step()
    return last_loss
