# Roger Federated — project brief

## What this is
- Local, sovereign agent: the user picks an open-source HF model; inference *and*
  finetuning run entirely on the user's own machine — data is never shared.
- Agentic rollouts (an agent completing a task begin-to-end) are recorded and used to
  finetune the model locally with LoRA RL.
- The resulting differentially-private gradients (not the data) are aggregated across a
  federation of users and broadcast back — federated learning. Over time the foundation
  model becomes purpose-trained for agency, rather than just an LLM with tools bolted on.

## Current state (read before assuming)
- Working semi-basic agent: rollout loop, tool use, reasoning, MCP connections, standard
  tools, deferred tool loading, auto-triggered RAG, skills + instruction files, `@path`
  references, persistent memory, and a CLI app (`roger`).
- The reward interface is in place (`training/reward_utils.py`). Conversation resume after `finish`
  is built: one rollout owns the session, awaiting each user turn and injecting it onto the live
  context so the KV-cache is reused across tasks. NOT yet built: the LoRA RL trainer, federated
  hives, docker sandboxing, and account/hive/scheduling setup. The authoritative TODO list is in
  `readme.md` ("Progress & contributing").

## Confirmed design decisions
- RL algorithm = **REINFORCE++**, not GRPO. Flat episode return broadcast over all generated
  tokens; batch-mean baseline; no KL term (LoRA already bounds drift); no env reset / no
  grouping of comparable rollouts (deliberately avoided — infeasible across federated users).
- **No state-embedding injection.** Observations / `get_state` are plain *text* tool results.
  Injected `state_embeds` have no token IDs, so their log-probs can't be recomputed during
  the policy-gradient update — text is the model's native, RL-safe interface.
- Rewards = implicit user signals + verifiable signals via `auto_signal` (nonzero exit codes,
  error strings, rejections), plus the model's own `finish(score=...)` self-evaluation in [-1,1]
  as each task's terminal reward (broadcast over that task's steps). No external LLM-as-judge.
- Constrained decoding via lm-format-enforcer with a name-enum schema (prevents misspelled
  tool names). Raw *pre-constraint* logits plus per-token allowed-set masks are recorded so
  the trainer can recompute constrained log-probs.

## Package layout (src-layout; package = `roger`, console script = `roger`)
- `agency/`   — rollout loop, tool/skill loaders, RAG retrieval, `@path` expansion
                (`rollout_utils`, `retrieval`, `skill_utils`, `path_utils`)
- `apps/`     — CLI entry point, config, Rich/prompt_toolkit UI (`cli`, `config`, `ui`)
- `serving/`  — model loading + VRAM-aware quantization tier selection (`model_setup`)
- `tools/`    — standard tools, shell execution + policy guardrails, MCP bridge
                (`std_tools`, `shell_tools`, `mcp_utils`, `command_policy.txt`)
- `training/` — RL machinery: reward shaping, trajectory recording, LoRA adapter + REINFORCE++
                trainer (`reward_utils`, `recording`, `lora_utils`, `trainer`)
- `federated/`— placeholder for gradient aggregation / differential privacy / strategies (empty)
- `envs/`     — not created yet (concrete shell/browser/code environments are future work)
- `tests/`    — e.g. `tests/test_rewards.py`

Runtime artifacts all live under the global `~/.roger/` (never in the project): `config.json`,
global `memory/memory.md` + per-project `memory/<dashed-abspath>.md`, `runs/`, `backups/`,
`scratch/`, `history`, and user `skills/`. Project-level skill dirs (`.agents/`, `.claude/`)
and instruction files (`AGENTS.md`/`CLAUDE.md`) are still read from the project. `build/` and
`*.egg-info/` are build output — ignore them; edit only under `src/`.

## Dev environment
- Python: use the conda env **`roger`** (Python 3.13, CUDA torch 2.12.0+cu130) —
  `conda run -n roger python ...`. It has the project installed editable (`pip install -e ".[audio]"`)
  plus pytest, so it covers syntax/import/test checks *and* real CUDA model loads. Bare
  `python`/`python3` hit the Windows Store stub (exit 49).
- No manual env patching needed: lm-format-enforcer 0.11.3 imports `PreTrainedTokenizerBase` from
  `transformers.tokenization_utils`, which transformers>=5.11 removed — a compat shim in
  `agency/rollout_utils.py` re-exposes it before the integration import, so any install just works.
- Default model `google/gemma-4-E2B-it` is **5.12B** params (not 2B); dev box GPU =
  RTX 1000 Ada, 6.44 GB VRAM (bf16 supported).
- Install/run for end users: `uv tool install . --torch-backend auto` then `roger`; or
  `uvx --from . --torch-backend auto roger`. Tests: `conda run -n roger python -m pytest tests/`.

## Conventions
- Functional-first Python. Use a class only when isolated mutable state genuinely requires it
  (module-level globals would be worse). A class that only groups a namespace → make it a module.
- Implement the minimum changes necessary; no speculative abstraction or future-proofing.
- Write plenty of dense, *why*-not-*what* inline comments; no docstrings that merely restate
  the signature.
- Don't mark a function `async` unless it actually `await`s something.
- Prefer a self-derived probe over a hardcoded list / magic string (e.g. chat-template probes
  that work for any model, rather than a maintained list of known tag pairs).
- Don't delete commands or code unrelated to your change. When asked to commit, group changes
  by relevance and commit separately.
- After completing a task, write the important aspects of your implementation and the
  interaction to your memory.

## Keep this file current
- When you find that something here is wrong, stale, or missing — and knowing it would help a
  future session — update `CLAUDE.md` as part of your work. But also in this regard, only edit if necessary. Memory should be written to your memory files; not to `CLAUDE.md`.
