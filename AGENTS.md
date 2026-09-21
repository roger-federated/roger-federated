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
- **Paradigm change (2026-09): roger is a wrapper around local runtime providers**, not its own
  harness/runtime. `roger <runtime-command> …` (e.g. `roger llama-server -m x.gguf --port 8080`,
  `roger vllm serve … --port 8000`) runs the command untouched; `runtime/dialect.py` + `runtime/proxy.py` move the server to
  an ephemeral loopback port (rewrites `--port`) and listens on the user's port as a transparent
  reverse proxy, and `runtime/capture.py` saves every 2xx `/v1/chat/completions` | `/v1/completions` |
  `/v1/responses` exchange as one JSON file under `~/.roger/messages/<runtime>/` (streamed SSE replies
  reassembled into the non-streaming object; request verbatim). **Only the standard surface is
  supported**: a `--port` flag on the runtime and OpenAI-style JSON/SSE on the wire. Deliberately no
  plumbing for non-standard runtimes — ollama (env-configured port, native NDJSON `/api/chat`) and
  LM Studio (`lms server start` detaches) are not supported for now. A command without `--port` just
  runs as-is (one stderr notice). **Supported-model notice** (`runtime/notice.py`, daemon thread from the
  wrapper): once the runtime answers `GET /v1/models` (the only standard way to learn the model — argv is
  runtime-specific), each federation's `/status` is probed and one stderr line per federation says whether
  it trains that model, and if not which it accepts (server advertises its allowlist as `models`, null =
  any); runtime ids (gguf paths, aliases) are matched to accepted `org/name` ids by normalised [a-z0-9]
  containment of `name`, longest wins. Fail-soft silent when a federation is unreachable; also carries the
  version verdicts (outdated client / update available) and the "contribute is off" nudge. `notice.privacy_notice`
  is the one-time transparency notice (sentinel `~/.roger/privacy_ack`, printed from `adapter.prepare`
  right before the day's pull = this client's first federation contact; the obligation `CLIENT_VERSION`
  2 was bumped to force-adopt). `transport._base` defaults a scheme-less federation URL (the shipped
  default is a bare host) to https. Console script = `runtime/wrapper.py:main`; **bare `roger` prints
  usage** and that is the only non-runtime invocation. **Daily pull + adapter**
  (`runtime/adapter.py`, before the child spawns): the model named on argv (`dialect.RUNTIMES` table:
  llama-server `-m`, vllm `serve <id>`/`--model`) is resolved against `/status` `models`, the global is
  pulled on the first launch of the UTC day (state + blob keyed per federation *and* served model), and
  every launch rebuilds a LoRA adapter from disk — GGUF (`--lora`, alpha=0 ⇒ scale 1, arch/shapes from the
  base gguf header, llama q-row permute) or PEFT dir (vllm `--enable-lora --lora-modules roger=… --max-lora-rank`,
  proxy rewrites chat requests' `model` to `roger`). Torch-free. Three command lines rule the update
  out by themselves and warn on stderr rather than skipping silently (the relay, capture and grading work
  regardless, so the session otherwise looks healthy): a runtime outside `RUNTIMES`, one already loading
  the user's own LoRA (`dialect.user_adapter`; roger neither stacks onto it nor overwrites it), and one
  naming no model at all (llama-server's `-hf`, a vllm config file). The first and last also mean
  `served_model` is null on every capture, so those chats can never be trained on. It expects the broadcast in **LoRA-factor
  form** (contract in `federated/delta.py` docstring), which `roger-server` now serves. **Automatic
  training round** (`runtime/train.py` gate, `training/wire_trainer.py` torch half): after the runtime exits
  (VRAM free), once ≥ `train_every` graded conversations for the served model exist (newest capture file of
  each chat, deduped by prefix; captures carry `served_model`), it trains in the foreground (Ctrl-C skips) on
  the model loaded *without download* — llama-server's GGUF via transformers `gguf_file=` (gemma-4 needs a
  transformers with the gemma4 GGUF processor: on main, not in 5.17.0), vllm's HF cache `local_files_only` —
  and uploads to exactly ONE federation (first in `federations` order that trains the model; only its global is
  attached for training, at scale 1 — while inference attaches all globals concatenated, each at 1/N); accepted ⇒ the conversation files (+ superseded prefixes) are
  deleted. **Factor contract (RoLoRA, mirrors roger-server):** each epoch trains one factor (`/status`
  `phase`) against the other frozen; the trained adapter IS the global (or derived `init_A`, B=0 cold),
  per-module rank `delta.rank_for`, canonical module keys `base_model.model.model.layers.N…` (text decoder
  only), upload = the phase factor's Δ stamped `base`/`epoch` (`client.contribute_factor`: masked, or
  DP-noised on the factor in bootstrap), `secure_agg.SCALE` = 2²². Episodes: chat template re-rendered, each
  assistant turn located by a prefix probe; return = mean(self_eval scores) + mean(self_eval reactions) + Σ tool_signals; old log-probs
  recomputed (no-grad) since the wire has none.
- **Self-evaluation over the wire (`runtime/grader.py`).** The reward source is the model's own
  end-of-session grade (a reason-then-force prefill: let it reason, then force the schema). The wrapper chains
  exchanges into conversations by message-prefix (in-memory registry, `grader.track`; every request
  carries the full history, so the previous exchange's transcript is a prefix of the next). Once a
  conversation is idle `IDLE_S` (5 min) — or at Ctrl-C, before the runtime is stopped — one eval call
  goes to the **backend port directly** (never relayed, never captured; under the relay's
  `request_model` when set, so a vllm adapter grades its own work): the transcript + a final
  assistant message holding the seed (prefill), `response_format` json_schema with properties ordered
  `reasoning` then `scores` (reason-then-force, over the API this time), no `tools`. Per-metric scores
  (`METRICS`: efficiency/accuracy/completeness, each clamped to [-1,1]) are persisted as `self_eval`
  on the conversation's newest file; **no aggregate is stored** — the optimiser averages. The same call
  also classifies the **user's reactions** — the wire's only human signal, since roger can't prompt
  inside a third-party client: each assistant message the user directly answered is
  quoted in the seed (`REACTION_SEED`, snippet of the answer) and forced as a `reactions.reply_N` property
  after `scores`, judged from the user's words only; stored as `self_eval.reactions` {message index:
  [-1,1]} (same keys as `tool_signals`). No answered replies ⇒ the bare SEED/SCHEMA. One generic
  fallback: a 4xx (alternation-strict templates reject two assistant messages) retries once with the
  seed appended to the last reply. Failures land as `self_eval.error`, never retried. Because of the
  Ctrl-C ordering the runtime child runs in its **own process group** (`start_new_session` /
  `CREATE_NEW_PROCESS_GROUP`) and gets the interrupt from roger after grading; a second Ctrl-C skips.
- **Tool-result signals over the wire (`runtime/signals.py`).** The verifiable per-step reward: the
  client runs the tools, so `capture` scores the `role: "tool"` / `function_call_output`
  items in each request's history (nonzero exit code in any common phrasing, error strings, "Command
  rejected by user") and stores `tool_signals` = {assistant-turn index: clamped sum}, nonzero
  steps only, on every file — the conversation's newest file therefore holds them all, next to `self_eval`.
- **The in-process harness is gone (2026-09).** `agency/` (rollout loop, tools, reasoning, MCP, RAG,
  skills, `@path`, memory, subagents, `/perpetual`), `apps/` (the Rich/prompt_toolkit REPL CLI), `tools/`,
  `loading/` (the VRAM-aware loader + rollback KV cache) and `skills/` were removed along with the
  `roger train` subcommand that was their last entry point. Don't look for them, and don't reintroduce a
  tool set or an agent loop: the user's client owns both, and everything roger needs arrives on the wire.
  What survived the move out of those packages: `paths.state_dir` (was `agency/path_utils`), `config`
  (was `apps/config`), the REINFORCE++ core in `training/trainer.py` (the run-dir plumbing, the
  10%-user-graded gate and `discard_runs` went with the runs they read), and the privacy notice (was
  `cli._ensure_privacy_notice`, now `notice.privacy_notice`). Git history has the rest if it is ever needed.
- LoRA REINFORCE++ is `training/trainer.py` (advantages, PII anonymisation, the teacher-forced log-prob
  pass, the clipped step) driven by `training/wire_trainer.py` (the round over captured chats); train-time
  PII anonymisation via `privacy_filter.py`, which now loads its detector straight from transformers.
- Federated gradient-sharing **client** (`federated/`): a round trains the federation's own global
  adapter on the **fixed basis q_proj/v_proj** (`lora_utils.FED_TARGETS`, NOT all-linear — it is the
  shared secure-agg layout, so it bounds the server's per-round work) and uploads ONE factor's Δ,
  masked with Bonawitz secure aggregation (X25519 EC-DH), to ONE federation (`client.contribute_factor`;
  it is the only entry point left). The server broadcasts the cumulative global in **LoRA-factor form**;
  the client pulls it daily and persists the blob under `~/.roger/federated/` (`transport`, keyed per
  federation *and* served model), and `runtime/adapter.py` — torch-free — rebuilds it as the runtime's
  own adapter. Nothing is ever folded into weights and no model copy is stored. Config: `contribute` /
  `federations` / `train_every` (the L2 clip is a fixed best-effort client constant, not user config —
  authoritative norm-bounding is server-side). Leech mode (config'd-in but not contributing) is nudged,
  not blocked.
- The federated aggregation **server now lives in a separate repo** (`roger-server`,
  github.com/roger-federated/roger-server; package `roger_server`, run `python -m roger_server`). It is
  wire-compatible with this client purely over HTTP: it seals secure-agg cohorts, streams + sums the
  masked uploads one factor at a time (masks cancel), folds η·mean(Δ) into a per-model cumulative global
  LoRA adapter, and broadcasts it; default deploy is a scale-to-zero container with the global in S3. The
  secure-aggregation + factor wire format is a **contract shared by hand across the two repos**:
  `federated/secure_agg.py` (SCALE/R/quantize/mask; `dequantize` is server-only and lives over there) and
  `federated/delta.py` here mirror `roger_server/secure_agg.py` + `roger_server/delta.py`. Any change to
  the quantization, the sorted-key flatten layout, the safetensors metadata keys, or an endpoint shape
  must be made on **both** sides.
- **Cold-start fix — DP-noised async bootstrap.** A sparse federation can't seal cohorts (needs k_min
  registrants in one ~20s window), so per model the server advertises a mode at `GET /status`: while
  sparse it serves `bootstrap` and clients skip the cohort entirely, uploading ONE **faux-DP-noised,
  unmasked** factor Δ to `POST /contribute_dp` that the server folds (k=1, `dp_fold`, same per-factor stream).
  Noise goes straight on the trained factor (σ=z·rms(Δ); client `DP_Z` in `client.contribute_factor`) —
  under the factor contract the frozen factor is public and Δ ↦ Δ·A is linear, so the weight-space noise
  is exactly Gaussian, unlike the old dense scheme. It's still *faux*-DP: fixed σ, no accountant —
  obfuscation against an honest-but-curious server, sound only because it's temporary. Once
  `busy_threshold` distinct contributors appear within `busy_window`, the model flips to **busy** mode
  (secure-agg only, no DP); quorum raised to **k_min=3 / k_target=5**. `contribute_factor` takes the mode
  its caller read off `/status`; no client config. (Mode is "bootstrap"/"busy" — "busy" rather
  than "dense" to avoid clashing with the dense-matrix sense of ΔW.) `GET /status` also returns a third
  mode, **"unsupported"**, when the server's allowlist (`ROGER_AGG_MODELS`) excludes the model (plus
  `models` = the allowlist itself, null when any model is accepted — part of the shared wire contract): the
  client skips that federation, `notice.announce` says so once the runtime is up (before a session's chats
  are wasted) and `train._federation` passes it over at training time, naming it with its reason and
  keeping its conversations. Every `mode` read fail-soft-defaults to "busy", so an unreachable server is
  never mistaken for an unsupported model.
- NOT yet built (see `readme.md` TODO + the federated-server-roadmap memory): the **server-side**
  roadmap — Shamir/double-mask dropout recovery (needs a client protocol change too; multi-round is
  intrinsic), central ground-truth-gradient anti-poison gate — now lives in the `roger-server` repo.
  Client/app side: docker sandboxing, account/hive/scheduling setup. (Membership auth is built: a
  secret token issued at `/round/register`, echoed back and checked at `/contribute`.)

## Confirmed design decisions
- RL algorithm = **REINFORCE++**, not GRPO. Flat episode return broadcast over all generated
  tokens; batch-mean baseline; no KL term (LoRA already bounds drift); no env reset / no
  grouping of comparable rollouts (deliberately avoided — infeasible across federated users).
- **Text only, never embeddings.** Everything the policy is trained on must have token IDs, or its
  log-probs can't be recomputed during the policy-gradient update. That ruled out injected state
  embeddings in the harness and it rules out anything but the rendered transcript now.
- Rewards = the model's own self-evaluation in [-1,1] per metric (the terminal reward, broadcast over the
  conversation's steps) + the user's reactions to the replies they answered + verifiable per-step signals
  read off the tool results (nonzero exit codes, error strings, rejections). No external LLM-as-judge.
- **Nothing is reconstructed that the wire can carry.** The behaviour log-probs are recomputed in one
  no-grad pass at the round's starting weights rather than guessed at, and each assistant turn's token
  span is found by a prefix probe against the model's own chat template rather than by hardcoded tags.

## Package layout (src-layout; package = `roger`, console script = `roger`)
- `paths.py`  — `state_dir()`: the one global `~/.roger` location everything writes under
- `config.py` — `~/.roger/config.json` (+ the shipped `config.json` defaults as package data)
- `runtime/`  — the provider wrapper (the entry point): `wrapper` (console script `main`, process
                lifecycle, Ctrl-C), `dialect` (**all runtime/framework-specific code lives here and only
                here**: `plan` = `--port` argv rewrite, `RUNTIMES` table of model flags, adapter
                args and the user's own LoRA flags, GGUF/PEFT adapter writers), `proxy` (runtime-agnostic stdlib `ThreadingHTTPServer`
                reverse proxy over a shared `httpx.Client`, streaming relay), `capture` (chat-path
                filter, SSE→non-stream reassembly, atomic JSON writes + `update_record` to `messages/`),
                `notice` (runtime `/v1/models` → federation `/status` supported-model verdicts on stderr,
                plus the one-time `privacy_notice`),
                `adapter` (daily global pull → LoRA adapter written + attached via `dialect.RUNTIMES`),
                `grader` (prefix-chain conversation registry, idle/shutdown self-eval call, `self_eval`),
                `signals` (exit-code/error step rewards from tool results → `tool_signals`),
                `train` (post-exit training gate: ready conversations, one federation, upload, delete)
- `training/` — the torch half, imported only once a round is actually due: `trainer` (the REINFORCE++
                algorithm: advantages, `anonymize`, the teacher-forced log-prob pass, the clipped step),
                `wire_trainer` (the round itself: load the served model, rebuild episodes from captured
                chats, attach the global, return the factor Δ), `lora_utils` (`attach_lora` + the fixed
                federation basis `FED_TARGETS`), `privacy_filter` (train-time PII → surrogates)
- `federated/`— gradient-sharing client: `delta` (the factor/rank/`init_A` contract + safetensors
                (de)serialization), `secure_agg` (X25519/EC-DH + SHAKE pairwise masks, quantize mod R),
                `transport` (httpx per-federation, fail-soft; `/status`, `/round/register`,
                `/contribute`, `/contribute_dp`, `/global` + sync state and the persisted global blob),
                `client` (`contribute_factor`: the masked or DP-noised upload to ONE federation). The
                pull half is torch-free and lives in `runtime/adapter.py`.
                The aggregation **server is a separate repo** (`roger-server`); this package is client-only.
                `secure_agg` + `delta` here mirror the server's copies of the wire contract (the server's
                `secure_agg` additionally carries the server-only `dequantize` half, and its `delta` a
                `base_from_json` the client never needs).
- `tests/`    — `test_runtime.py` (plan/proxy/capture/notice/adapter/grader), `test_wire_training.py`
                (contribute_factor, the conversation gate, the round), `test_federated.py` (the mirrored
                wire contract + secure-agg crypto), `test_trainer.py`, `test_privacy_filter.py`

Runtime artifacts all live under the global `~/.roger/` (never in the project): `config.json`,
`messages/<runtime>/` (the captured chats, text), `federated/` (per-federation-and-model sync state +
global blob + the built `adapters/`), and the `privacy_ack` sentinel. Nothing is written into the
project any more, and no model is ever stored. `build/` and `*.egg-info/` are build output — ignore
them; edit only under `src/`.

## Dev environment
- Python: use the conda env **`roger`** (Python 3.13, CUDA torch 2.12.0+cu130) —
  `conda run -n roger python ...`. It has the project installed editable (`pip install -e .`) plus
  pytest, so it covers syntax/import/test checks *and* real CUDA model loads. Bare `python`/`python3`
  hit the Windows Store stub (exit 49).
- Roger no longer has a model of its own: the model is whatever the user's runtime command names, and
  the training round reads it from that file or cache without downloading. The config has no `model_id`.
- The default federation only accepts gemma-4 bases (any quantization of `google/gemma-4-12B-it` or
  `google/gemma-4-E2B-it`). Dev box GPU = RTX 1000 Ada, 6.44 GB VRAM (bf16 supported), so the 12B
  won't fit there — test a round against E2B locally. A gemma-4 GGUF needs a transformers with the
  gemma4 GGUF processor (on main, not in 5.17.0).
- Install/run for end users: `uv tool install . --torch-backend auto`, then prefix their runtime
  command with `roger`; or `uvx --from . --torch-backend auto roger <runtime-command>`.
  Tests: `conda run -n roger python -m pytest tests/` (CPU-only, download-free).

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
