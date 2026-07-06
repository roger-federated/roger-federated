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
  references, persistent memory, web search/fetch, sub-agent spawning, and a CLI app (`roger`).
- Perpetual standing tasks (`/perpetual <task>`): at the done-boundary the rollout re-seeds a varied
  (randomly-rotated, comprehensive) think-channel continuation nudge instead of yielding to the user,
  grading+checkpointing each iteration into the one session trajectory; `prompt_user` is grammar-
  suppressed and the max-steps check-in bypassed so the agent can't yield. A graceful **Ctrl-C**
  (SIGINT → stop flag → `StoppingCriteria`) is the only stop and works for every rollout (wraps up at
  the next boundary, back to the prompt). Seeds share a "reason-then-force" helper (open think, let
  the model close its own thought, then force the mandatory call) also used by the grade/memory nudges.
- LoRA REINFORCE++ trainer is built and wired (`training/trainer.py`, `lora_utils.py`); train-time
  PII anonymisation via `privacy_filter.py`. Gated Ctrl-D auto-train + `roger train` subcommand.
  `/grade` user override of `finish()` self-eval score + 10%-user-graded training gate.
- Federated gradient-sharing **client** is built (`federated/`): a training round trains a single
  fresh LoRA adapter on the **fixed federation basis q_proj/v_proj** (`lora_utils.FED_TARGETS`, NOT
  all-linear — it is the shared secure-agg dense basis, so it bounds the server's per-round work),
  exports its weight-space ΔW (=scaling·B@A) — never applied/saved locally —
  densifies + masks it with Bonawitz secure aggregation (X25519 EC-DH), and uploads per federation.
  The server broadcasts the **full cumulative dense global** ΔW; the client pulls it daily, persists
  the blob under `~/.roger/federated/`, and **folds it into the base in bf16 at load, then bnb-quantizes
  to GPU** (`delta.fold_into` + `model_setup.fetch_model(weight_deltas=…)`) — the HF cache is untouched
  and no model is ever stored. Config: `contribute`/`federations` (the ΔW L2 clip is a fixed
  best-effort client constant, not user config — authoritative norm-bounding is server-side). Leech
  mode (config'd-in but not contributing) is nudged, not blocked.
- The federated aggregation **server now lives in a separate repo** (`roger-server`,
  github.com/roger-federated/roger-server; package `roger_server`, run `python -m roger_server`). It is
  wire-compatible with this client purely over HTTP: it seals secure-agg cohorts, streams + sums the
  masked uploads one module at a time (masks cancel), folds η·mean(ΔW) into a per-model cumulative dense
  global, and broadcasts it; default deploy is a scale-to-zero container with the global in S3. The
  secure-aggregation + ΔW wire format is a **contract shared by hand across the two repos**:
  `federated/secure_agg.py` (SCALE/R/quantize/mask; `dequantize` is server-only and lives over there) and
  `federated/delta.py` here mirror `roger_server/secure_agg.py` + `roger_server/delta.py`. Any change to
  the quantization, the sorted-key flatten layout, the safetensors metadata keys, or an endpoint shape
  must be made on **both** sides.
- **Cold-start fix — DP-noised async bootstrap.** A sparse federation can't seal cohorts (needs k_min
  registrants in one ~20s window), so per model the server advertises a mode at `GET /status`: while
  sparse it serves `bootstrap` and clients skip the cohort entirely, uploading ONE **faux-DP-noised,
  unmasked** dense ΔW to `POST /contribute_dp` that the server folds (k=1, `dp_fold`, same per-module stream).
  Noise is injected in **LoRA-factor space before densifying** (`delta._dp_noise`, per-factor
  σ=z·rms; client `DP_Z`) — the rank-r signal subspace, far less SNR loss than noising the dense matrix.
  It's *faux*-DP: fixed σ, no accountant, and B@A is bilinear so not a formal Gaussian mechanism —
  obfuscation against an honest-but-curious server, sound only because it's temporary. Once
  `busy_threshold` distinct contributors appear within `busy_window`, the model flips to **busy** mode
  (secure-agg only, no DP); quorum raised to **k_min=3 / k_target=5**. Client picks the path per
  federation in `contribute_delta`; no new client config. (Mode is "bootstrap"/"busy" — "busy" rather
  than "dense" to avoid clashing with the dense-matrix sense of ΔW.) `GET /status` also returns a third
  mode, **"unsupported"**, when the server's allowlist (`ROGER_AGG_MODELS`) excludes the model: the
  client then skips that federation and the CLI warns the user (at startup before a session's gradient is
  wasted, and again at quit, where it skips training but KEEPS the recorded runs). `federation_mode`
  fail-soft-defaults to "busy", so an unreachable server is never mistaken for an unsupported model.
- NOT yet built (see `readme.md` TODO + the federated-server-roadmap memory): the **server-side**
  roadmap — Shamir/double-mask dropout recovery (needs a client protocol change too; multi-round is
  intrinsic), central ground-truth-gradient anti-poison gate — now lives in the `roger-server` repo.
  Client/app side: docker sandboxing, account/hive/scheduling setup. (Membership auth is built: a
  secret token issued at `/round/register`, echoed back and checked at `/contribute`.)

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
                (`rollout_utils`, `retrieval`, `skill_utils`, `path_utils`), sub-agent
                spawning (`subagents`)
- `apps/`     — CLI entry point, config, Rich/prompt_toolkit UI (`cli`, `config`, `ui`)
- `loading/`  — model loading + VRAM-aware quantization tier selection (`model_setup`),
                rollback sliding-window KV cache (`rollback_cache`)
- `tools/`    — standard tools, shell execution + policy guardrails, MCP bridge
                (`std_tools`, `shell_tools`, `mcp_utils`, `command_policy.txt`)
- `training/` — RL machinery: reward shaping, trajectory recording, LoRA adapter + REINFORCE++
                trainer, train-time PII anonymizer
                (`reward_utils`, `recording`, `lora_utils`, `trainer`, `privacy_filter`)
- `skills/`   — bundled default skills shipped as package-data (`ipynb`, `skill-creator`,
                `git-workflow`, `code`, `lean`); read in place as the lowest-priority `discover_skills` base
- `federated/`— gradient-sharing client: `delta` (densify ΔW [+ optional factor-space DP noise] +
                (de)serialize + `fold_into` the base weights in bf16), `secure_agg` (X25519/EC-DH +
                SHAKE pairwise masks, quantize mod R), `transport` (httpx per-federation, fail-soft,
                `federation_mode`/`contribute_dp` + sync state + persisted global blob), `client`
                (mode-branched contribute / daily-pull / `pending_globals` / leech gating). The
                bf16-fold-then-bnb-quantize loading lives in `loading/model_setup.fetch_model(weight_deltas=…)`.
                The aggregation **server is a separate repo** (`roger-server`); this package is client-only.
                `secure_agg` + `delta` here mirror the server's copies of the wire contract (the server's
                `secure_agg` additionally carries the server-only `dequantize` half).
- `envs/`     — not created yet (concrete shell/browser/code environments are future work)
- `tests/`    — `test_rewards.py`, `test_trainer.py`, `test_grade.py`, `test_privacy_filter.py`,
                `test_mcp.py`, `test_multimodal.py`, `test_federated.py`

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
- Shipped default model is `google/gemma-4-12B-it` with `google/gemma-4-12B-it-assistant`
  (1B despite the name) as the default speculative-decoding drafter. The default federation only
  accepts gemma-4 bases. The small `google/gemma-4-E2B-it` (**5.12B** params, not 2B) is the
  low-VRAM experimentation fallback; dev box GPU = RTX 1000 Ada, 6.44 GB VRAM (bf16 supported), so
  the 12B default won't fit there — test against E2B locally.
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
