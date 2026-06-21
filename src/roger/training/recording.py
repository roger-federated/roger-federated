"""recording.py — Persist rollout trajectories for downstream LoRA RL training.

Each run is saved to:
  ~/.roger/runs/<ISO-timestamp>/
    trajectory.pt       — torch.save of per-step dicts (gen_start, old_logp,
                          masks, reward; optional input_mm = mm kwargs dict
                          (pixel_values, token_type_ids, …) when that turn's
                          input carried an image); consumed by the REINFORCE++ trainer.
    sequence.pt         — the cumulative token-id sequence for the episode; the trainer slices
                          each turn's span [gen_start, gen_start+len) out of it to teacher-force
                          a differentiable forward pass.
    transcript.jsonl    — human-readable prompt + per-step events.

Public API:
  save_run(trajectory, prompt, run_dir=None, seq_ids=None) → str  (returns the run directory path)
"""
import json, os
from datetime import datetime, timezone

import torch

from rich.console import Console

from roger.agency.path_utils import state_dir

console = Console(highlight=False)


def save_run(trajectory: list, prompt: str, run_dir: str | None = None,
             seq_ids: torch.Tensor | None = None) -> str:
    """Save trajectory to disk. Reuse `run_dir` if given (a continuous session checkpoints the
    same growing episode on every finish); otherwise create a fresh ~/.roger/runs/<timestamp>/
    and return it so the caller can keep checkpointing into it."""
    if not trajectory:
        return run_dir or ""  # nothing to record yet

    if run_dir is None:
        ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_dir = os.path.join(state_dir(), "runs", ts)
    os.makedirs(run_dir, exist_ok=True)

    # --- trajectory.pt: per-step dicts (RL training input) ---
    # Each step has gen_start (int), old_logp (tensor), masks (list), reward (float)
    torch.save(trajectory, os.path.join(run_dir, "trajectory.pt"))
    # --- sequence.pt: the cumulative token ids the trainer teacher-forces over ---
    # Re-saved (overwritten) on each checkpoint; the latest is a superset covering every gen_start.
    if seq_ids is not None:
        torch.save(seq_ids, os.path.join(run_dir, "sequence.pt"))

    # --- transcript.jsonl: human-readable, one JSON object per line ---
    transcript_path = os.path.join(run_dir, "transcript.jsonl")
    with open(transcript_path, "w", encoding="utf-8") as f:
        # Header line: the user's prompt
        f.write(json.dumps({"type": "prompt", "text": prompt}) + "\n")
        for i, step in enumerate(trajectory):
            entry = {"type": "step", "step": i, "reward": float(step.get("reward", 0))}
            if "input_mm" in step:
                entry["has_image"] = True
            # One mask per generated token, so its length is the turn's generated-token count.
            masks = step.get("masks")
            if masks is not None:
                entry["gen_len"] = len(masks)
            f.write(json.dumps(entry) + "\n")

    console.print(f"[dim]Run saved → {run_dir}[/dim]")
    return run_dir
