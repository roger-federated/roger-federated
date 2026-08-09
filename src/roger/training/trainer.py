"""trainer.py — local LoRA REINFORCE++ update over recorded rollouts.

Teacher-forces the LoRA policy over each recorded episode to recompute differentiable per-token
log-probs, then applies a flat REINFORCE++ step: episode return over all its tokens, batch-mean
baseline, z-normed advantages, PPO-clipped ratio vs the stored behaviour log-probs, no KL. The
adapter is saved to ~/.roger/adapter and consumed runs are deleted. Wants fresh (on-policy) data.
"""
import glob
import math
import os
import shutil
from contextlib import nullcontext

import torch

from roger.agency.path_utils import state_dir
from roger.training import lora_utils, privacy_filter


def _runs_dir() -> str:
    return os.path.join(state_dir(), "runs")


def _list_unconsumed(limit: int) -> list[str]:
    """Most-recent `limit` complete run dirs. Consumed runs are deleted outright, so 'complete and
    still present' == unconsumed. Newest favoured for on-policy freshness."""
    runs = sorted(glob.glob(os.path.join(_runs_dir(), "*")))   # ISO-timestamp names sort chronologically
    ready = [d for d in runs
             if os.path.exists(os.path.join(d, "trajectory.pt"))
             and os.path.exists(os.path.join(d, "sequence.pt"))]
    return ready[-limit:]


def user_grade_shortfall(batch: int = 8) -> int:
    """How many more user-graded runs the next pass needs to clear the 10% rule (0 when satisfied
    or too few runs to train). Stat-only, so cheap enough to call at every prompt for the nudge.
    A run is user-graded iff the rollout dropped a `user_graded` sentinel on a /grade override."""
    dirs = _list_unconsumed(batch)
    if len(dirs) < 2:                      # a pass needs >= 2 episodes anyway; nothing to nudge yet
        return 0
    n_user = sum(os.path.exists(os.path.join(d, "user_graded")) for d in dirs)
    return max(0, max(1, math.ceil(0.10 * len(dirs))) - n_user)   # 10%-or-at-least-one


def _load_episode(run_dir: str):
    """Load a run into {dir, traj, seq}, or a flag if it is multimodal (not yet supported)."""
    # weights_only=True: tensors + plain containers only; closes the pickle-RCE vector for peer runs.
    traj = torch.load(os.path.join(run_dir, "trajectory.pt"), map_location="cpu", weights_only=True)
    seq  = torch.load(os.path.join(run_dir, "sequence.pt"),   map_location="cpu", weights_only=True).long()
    if not traj:
        return None
    if any("input_mm" in e for e in traj):
        return "mm"   # a text-only forward would mis-embed image tokens; deferred
    return {"dir": run_dir, "traj": traj, "seq": seq}


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


def train(model_id: str | None = None, *, batch: int = 8, epochs: int = 1, lr: float = 1e-5,
          clip_eps: float = 0.2, max_grad_norm: float = 1.0, targets=lora_utils.FED_TARGETS,
          reuse=None) -> dict:
    """Run one REINFORCE++ update over up to `batch` unconsumed runs. `reuse=(model, processor)`
    trains the already-loaded in-session model (Ctrl-D path); None loads its own via
    fetch_model(for_training=True). Returns a stats dict; never raises on 'nothing to do'."""
    import bitsandbytes as bnb
    from roger.loading.model_setup import fetch_model

    dirs = _list_unconsumed(batch)
    eps, skipped_mm = [], 0
    for d in dirs:
        ep = _load_episode(d)
        if ep == "mm":
            skipped_mm += 1
        elif ep is not None:
            eps.append(ep)
    # Batch-mean baseline + z-norm are undefined for a single sample; wait for more data.
    if len(eps) < 2:
        return {"trained": False, "reason": "need >= 2 episodes", "n_ready": len(eps), "skipped_mm": skipped_mm}
    # Require >=10% (and >=1) user-graded runs so the signal isn't purely the model grading itself.
    # Blocked passes delete nothing (deletion is post-update below), so runs wait for a graded batch.
    n_user, need = sum(os.path.exists(os.path.join(ep["dir"], "user_graded")) for ep in eps), max(1, math.ceil(0.10 * len(eps)))
    if n_user < need:
        return {"trained": False, "reason": "10% rule: too few user-graded runs",
                "n_user": n_user, "need": need, "n_ready": len(eps), "skipped_mm": skipped_mm}

    returns  = torch.tensor([sum(float(e["reward"]) for e in ep["traj"]) for ep in eps])
    centered = returns - returns.mean() # batch-mean baseline (REINFORCE++)
    if float(centered.std()) == 0.0:
        return {"trained": False, "reason": "zero-variance returns", "n_ready": len(eps), "skipped_mm": skipped_mm}
    adv = centered / centered.std().clamp_min(1e-6) # z-norm; one scalar per episode

    if reuse is not None:
        model, processor = reuse                # already loaded with the global folded (see cli._repl)
    else:
        # Fold the cumulative global into the base before the fresh adapter goes on top, exactly as
        # the REPL loads it. Without this the standalone `roger train` computes the round's update at
        # a stale point in weight space (every other member trains from base+global), and the recorded
        # old_logp — collected against the folded model — disagrees with the recomputed log-probs, so
        # the ratio leaves 1 and clipping bites over a base-weight difference rather than an update.
        from roger.apps import config
        from roger.federated import client as fed_client
        model, processor = fetch_model(model_id, for_training=True,
                                       weight_deltas=fed_client.pending_globals(config.load()))

    # Rewrite PII to surrogates before any gradient sees it; free the filter before training so it
    # doesn't hold VRAM. Detect/load failures propagate rather than train on raw PII.
    tokenizer = getattr(processor, "tokenizer", processor)
    for ep in eps:
        new_seq, pii_pos = privacy_filter.anonymize_sequence(ep["seq"], tokenizer)
        ep["seq"] = new_seq
        # keep mask, in _new_logps' per-step generated-token order; 0 = a rewritten PII token.
        keep = [0.0 if (int(e["gen_start"]) + j) in pii_pos else 1.0
                for e in ep["traj"] for j in range(len(e["masks"]))]
        ep["keep"] = torch.tensor(keep)
    privacy_filter.free_filter()

    # Trains a fresh `local` adapter on top of the frozen global; its ΔW is what we share.
    model = lora_utils.attach_lora(model, targets=targets)
    model.train()
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.Adam8bit(trainable, lr=lr)

    # 1/total_tokens scaling + per-episode backward() accumulation = exact token-mean, no padding.
    # total_tokens counts only kept tokens, so dropped PII positions don't skew the token-mean.
    gen_tokens   = sum(len(e["masks"]) for ep in eps for e in ep["traj"])
    total_tokens = int(sum(float(ep["keep"].sum()) for ep in eps))
    if total_tokens == 0: # every generated token was PII → nothing to learn from
        return {"trained": False, "reason": "all tokens masked", "n_ready": len(eps),
                "skipped_mm": skipped_mm}
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

    # No local apply: nothing is written to disk here. Instead export the adapter's factors (+ scaling)
    # as the federated contribution; the federated client densifies B@A into a weight-space ΔW, masks
    # it (secure aggregation), and uploads it. The local model only ever changes when the daily global
    # is pulled and folded into the base at load time (see federated/client.py + delta.fold_into).
    pc      = next(iter(model.peft_config.values()))   # the single adapter get_peft_model created
    delta   = {"weights":  lora_utils.local_state_dict(model),
               "scaling":  pc.lora_alpha / pc.r,
               "model_id": model_id}
    # Caller discards consumed dirs (via discard_runs) once the contribution is accepted
    return {"trained": True, "n_episodes": len(eps), "n_tokens": total_tokens,
            "n_pii_dropped": gen_tokens - total_tokens, "delta": delta,
            "consumed_dirs": [ep["dir"] for ep in eps],
            "mean_return": float(returns.mean()), "loss": last_loss, "skipped_mm": skipped_mm}


def discard_runs(dirs: list[str]) -> None:
    """Delete consumed run dirs once their ΔW has been contributed (on-policy, consume-once)."""
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
