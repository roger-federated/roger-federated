"""recording.py — Persist rollout trajectories for downstream LoRA RL training.

Each run is saved to:
  .roger/runs/<ISO-timestamp>/
    trajectory.pt       — torch.save of per-step tensors (gen_token_ids, logits,
                          masks, reward); consumed by the REINFORCE++ trainer.
    transcript.jsonl    — human-readable prompt + per-step events.

Public API:
  save_run(trajectory, prompt, root) → str  (returns the run directory path)
"""
import json, os
from datetime import datetime, timezone

import torch

from rich.console import Console
console = Console(highlight=False)


def save_run(trajectory: list, prompt: str, root: str) -> str:
    """Save trajectory to disk under <root>/.roger/runs/<timestamp>/."""
    if not trajectory:
        return ""  # nothing to record (e.g. cancelled before first step)

    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = os.path.join(root, ".roger", "runs", ts)
    os.makedirs(run_dir, exist_ok=True)

    # --- trajectory.pt: tensors only (RL training input) ---
    # Each step has gen_token_ids (tensor), logits (tensor|None), masks (tensor|None), reward (float)
    torch.save(trajectory, os.path.join(run_dir, "trajectory.pt"))

    # --- transcript.jsonl: human-readable, one JSON object per line ---
    transcript_path = os.path.join(run_dir, "transcript.jsonl")
    with open(transcript_path, "w", encoding="utf-8") as f:
        # Header line: the user's prompt
        f.write(json.dumps({"type": "prompt", "text": prompt}) + "\n")
        for i, step in enumerate(trajectory):
            entry = {"type": "step", "step": i, "reward": float(step.get("reward", 0))}
            # Include decoded token IDs as text if a tokenizer is not available here;
            # raw token IDs are more useful for debugging than silence.
            ids = step.get("gen_token_ids")
            if ids is not None:
                if hasattr(ids, "tolist"):
                    entry["gen_token_ids_len"] = len(ids.tolist())
                else:
                    entry["gen_token_ids_len"] = len(ids)
            f.write(json.dumps(entry) + "\n")

    console.print(f"[dim]Run saved → {run_dir}[/dim]")
    return run_dir
