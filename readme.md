![Roger Federated](assets/banner.PNG)

## Overview
A dichotomy in the AI transition is surfacing: Will we continue [to delegate our decisional capacity to closed source models we cannot inspect nor steer](https://blog.mozilla.org/en/mozilla/mozilla-open-source-ai-strategy/) and thus rely on [far-fetched financial projections](https://futurumgroup.com/insights/ai-capex-2026-the-690b-infrastructure-sprint/), or can we do better with [tiny models running on consumer hardware](https://newsletter.semianalysis.com/p/google-we-have-no-moat-and-neither)? *The answer is yes, we can do better.* And that's where Roger Federated comes in.

You and the community can now contribute to the next generation of AI. Not just an LLM with tools, but a purpose-trained agent that is inherently omni-modal. How? By *locally* finetuning a selected *open-source* foundation model on *agentic rollout data*, and subsequently *aggregating* the resulting encrypted model updates (not the data itself) securely with your selected *federations*.

![](assets/divider.PNG)

## Features
Roger is a wrapper around the local runtime you already use: keep your own client, your own model and your own agent, and prefix the server command with `roger`.

- **Nothing to switch to**: `roger llama-server …` / `roger vllm serve …` runs your runtime untouched and relays its port, so whatever client you chat with keeps working exactly as before.
- **Federated learning using SMPC**: Based on secure multi-party computing, only encrypted weight updates are contributed to your chosen federations. No peer or server can decipher the update and no raw data is ever shared. The aggregated global update comes back as a LoRA adapter your runtime attaches at load; your model file is never modified.
- **Efficient local reinforcement learning**: Inference and finetuning run entirely on the user's own machine; raw data never leaves it. QLoRA REINFORCE++ makes on-device RL efficient on consumer GPUs, and the round runs only once the runtime has exited and freed its VRAM.
- **Privacy filter**: Before any gradient is computed, a train-time anonymiser swaps personally identifiable information for consistent surrogates, so personal data can neither be learned nor transmitted.
- **Self-evaluation rewards**: Each conversation is scored by the model's own end-of-session self-evaluation, by how you reacted to its replies, and by verifiable signals read off the tool results in the transcript.
- **Scales to world models**: The same text-native, RL-safe interface extends from agency to world-models, which the federation can train collaboratively and deploy more cheaply than alternative centralized efforts.
- **More than just software**: Federations, continuous model updating, and an exchange of community-trained adapters make Roger a unique ecosystem that improves as more people contribute.

![](assets/divider.PNG)

## Installation

First, download the repo and navigate to the folder. For Windows users: install under WSL2 to use Flash Attention 2 and get the best performance.

```bash
git clone https://github.com/roger-federated/roger-federated.git # or extract from https://github.com/roger-federated/roger-federated/archive/refs/heads/main.zip
cd roger-federated
```

Note:

- Compatible GPU strongly recommended for the training round's speed. Roger never downloads a model: it trains the very file or cache your runtime served.
- First run writes settings under `~/.roger/config.json` (which federations to join, whether to contribute, how many graded chats a round needs). These can be changed at any time.

**Recommended method:**

Additionally requires [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to be installed (provisions Python and isolates dependencies).

```bash
uv tool install . --torch-backend auto   # installs `roger` globally; auto-picks CUDA/CPU torch
# or run without installing:
uvx --from . --torch-backend auto roger
```

<details>
<summary>Platform-dependent caveats</summary>

- `bitsandbytes` (4/8-bit quantization) is installed automatically only where PyPI ships a wheel: x86-64 Linux and Windows. On other CUDA platforms (e.g. aarch64 Jetson/GH200) install a custom wheel manually, e.g. `pip install --force-reinstall https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_aarch64.whl`.

- Apple Silicon: the training round falls back to MPS where available and to unquantized CPU otherwise; only the CUDA path gets QLoRA and 8-bit Adam (see `src/roger/training/wire_trainer.py`).
</details>

**Run:**

Roger wraps the local runtime you already use: prefix its usual command with `roger` and pass the arguments as normal. It relays the server's `--port` and saves every chat that flows through it under `~/.roger/messages/<runtime>/`.

```bash
roger llama-server -m model.gguf --port 8080
roger vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

Any OpenAI-compatible server that takes a `--port` flag works; runtimes with their own conventions (ollama's env-configured port, LM Studio's detached `lms server start`) are not supported yet. A command without an explicit `--port` still runs, but roger announces that it is inactive for that session (nothing is relayed or saved) and shows a supported invocation instead.

Once the runtime is up, roger asks it which model it serves (`GET /v1/models`) and tells you whether your federations train that model — and if not, which models they do accept, so you can switch. The default federation currently only accepts Gemma-4 bases (any quantization of `google/gemma-4-12B-it` or `google/gemma-4-E2B-it`); chats through an unsupported model are still saved, they just won't feed the federation.

On the first launch of each day, roger pulls your federations' latest model update for the model named on the command line (`-m` for llama-server, the served model for vllm) and keeps it under `~/.roger/federated/`. Every launch then attaches it as a LoRA adapter through the runtime's own flag (`--lora` for llama-server, `--enable-lora --lora-modules` for vllm, with chat requests routed to the adapter automatically), so your model file is never modified and no model copy is stored. llama-server needs `-m` to point at a local gguf for this.

Once the runtime exits, roger trains on what has piled up. A conversation counts once the model has graded it, which happens five minutes after you stop chatting or when you press Ctrl-C; once `train_every` graded conversations exist for that model (8 by default), the round runs in the foreground and uploads its update to one federation. Ctrl-C skips it and keeps the conversations for next time; they are deleted only once a federation has accepted the update.

<details>
<summary>Remote execution on a trusted machine over SSH</summary>

Roger installs and runs identically on any machine you can SSH into, so a rented GPU instance works exactly like your local one. We use Scaleway, but any provider works the same way, e.g. Hetzner, Koyeb, Stackit.

- Rent a GPU instance from a provider offering on-demand GPU compute.
- *Important*: Attach block storage that survives instance deletion, in order to persist the state that lives under `~/.roger/`. Alternatively, you could reverse-tunnel and symlink so that the state is mounted to your local machine.
- On the remote, install the same way, by cloning the repo and running `uv tool install . --torch-backend auto`.
- Optionally run from `tmux` so the session survives an SSH disconnect.

</details>

**Tools and MCP servers:**

Roger no longer runs the agent loop, so it has no tool set and no MCP configuration of its own: the client you chat with owns both. Configure your tools and MCP servers there, as you already do. Roger reads the tool results out of the transcript your client sends and turns them into training signal (a non-zero exit code, an error string or a rejected command all count against the step that caused them), so a richer tool set makes for a better gradient without any setup on roger's side.

![](assets/divider.PNG)

## Use cases
There is no limit to what a local agent can be delegated, and every such session is training data. The recordings below were made with roger's own in-process agent, before the pivot to wrapping the runtime; they show the kind of work the federation learns from, which is now done in whichever client you point at your `roger`-wrapped server.

<details>
<summary>Set up a native agent loop for a 24/7 unsupervised e-marketeer</summary>

[![Watch: 24/7 autonomous marketing agent](https://img.youtube.com/vi/8RU9dV9tiLc/0.jpg)](https://youtu.be/8RU9dV9tiLc)

</details>

<details>
<summary>Use always-on mode (with Touchpoint MCP) for assistance in e.g. music production</summary>
...
</details>

<details>
<summary>Automate any MS Office-based task</summary>
...
</details>

<details>
<summary>Develop e.g. a Steam game fully autonomously as a coding agent</summary>
...
</details>

![](assets/divider.PNG)

## Pricing

- **Individual**: Free. No strings attached.
- **Leechers**: Free for now. Per-pull charges may apply in the future.
- **Enterprise**: Free for now. In the future, pulling model-update may require a license depending on company size.

![](assets/divider.PNG)

## Progress & contributing
The ecosystem is still in development. Below is a non-exhaustive list of to-do items. Of course, we are an open-source community, so **feel free to open an issue or pull request!**

- [x] Investigate framework and open-source models.
- [x] Set up automatic model loading.
- [x] Write code to handle and record agentic rollouts using HF models.
- [x] Give the model agency, i.e., integrate tool use, reasoning, etc., with the LLM as foundation.
- [x] Enable connecting to MCP servers.
- [x] Implement standard tools (e.g., run_command, write_file, prompt_user).
- [x] Replace tool searcher with catalog+load_tools deferred loading (no embedding model needed).
- [x] Implement auto-triggered RAG.
- [x] Allow importing skills and agent.md files.
- [x] Add @path ability.
- [x] Integrate into a CLI application.
- [x] Prettify CLI application.
- [x] Auto read/write memory and create agent folder for temp files.
- [x] Integrate web search and fetch by default.
- [x] Allow resuming conversation after finish.
- [x] <ins>Whoopee, that's a semi-basic agent.</ins>
- [x] Implement what the finetuning framework considers as rewards.
- [x] LLM-as-judge (self-evaluation inspired by RLSR, SRT, Co-rewarding, meta-evaluation; requires a portion of ground truth).
- [x] Implement automatic QLoRA REINFORCE++.
- [x] Use privacy filter for training data.
- [x] Set up gradient contributing and Bonawitz secure aggregation.
- [x] Build the aggregation server: FedAvg, peer-key distribution, aggregate norm-bounding.
- [x] Use DP without accountant until server is busy, then switch to SMPC.
- [x] Launch the default server as a scale-to-zero container on Scaleway with S3 storage.
- [x] Use signatures and tokens (per-registration secret token proves cohort membership at upload).
- [x] <ins>Huzzah, the beta version can now be shipped.</ins>
- [x] Automatic subagent spawning (`spawn_subagent`) with concurrent tool dispatch.
- [x] Native agent loops: `/perpetual` standing tasks with graceful Ctrl-C stop.
- [x] Wrapper: daily pull of the federation update, attached to llama-server / vllm as a LoRA adapter.
- [x] Wrapper: capture chats off the wire, self-grade them, and train + contribute once the runtime exits.
- [x] <ins>Pivot complete: roger is a wrapper, not a harness.</ins> The in-process agent above (rollout loop, tools, MCP, RAG, skills, memory, subagents, `/perpetual`, the Rich CLI) has been removed - a third-party client does all of that better, and roger's job is the learning underneath it. The items above stay as a record of what was built and learned; they are not current features.

Deferred:
- [ ] Zero-knowledge integrity proof to verify scale, mod, keys, clip, model fork.
- [ ] Shamir dropout recovery.
- [ ] Sandboxed docker environments.
- [ ] Integrate spoken instructions.
- [ ] Remote control: copy a session code, enter it on our website, continue interacting encrypted through the browser.
- [ ] Always-on mode: always listen, look, and read, and when a keyboard shortcut is entered, automatically infer and continue the user's task using e.g. Touchpoint.
- [ ] Dynalang-style world model for embedded state roll-forward.
- [ ] Centrally coordinated ground-truth gradient injection.
- [ ] Automatic git worktrees for parallel subagent isolation.