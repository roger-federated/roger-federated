![Roger Federated](assets/banner.PNG)

## Overview
A fracture in the AI transition is surfacing: Will we continue [to delegate our decisional capacity to closed source models we cannot inspect nor steer](https://blog.mozilla.org/en/mozilla/mozilla-open-source-ai-strategy/) and thus rely on [far-fetched financial projections](https://futurumgroup.com/insights/ai-capex-2026-the-690b-infrastructure-sprint/), or can we do better with [tiny models running on consumer hardware](https://newsletter.semianalysis.com/p/google-we-have-no-moat-and-neither)? Indeed, *we can do better.* And that's why Roger Federated exists.

You and the community can now contribute to the next generation of AI. Not just an LLM with tools, but a purpose-trained agent. How? By *locally* finetuning a selected *open-source* foundation model on *agentic rollout data*, and subsequently *aggregating* the resulting encrypted model updates securely with your selected *federations*. As icing on the cake, you don't even have to change your workflow at all! Use your favourite model, your favourite runtime, and your favourite agent harness.

![](assets/divider.PNG)

## Features
On top of the basic capabilities you may expect, Roger adds the following.

- **Wrap around whichever runtime you already use**: llama.cpp and vLLM run untouched and roger relays its port, so whichever harness you use keeps working exactly as before.
- **Federated learning using SMPC**: Based on secure multi-party computing, only encrypted weight updates are contributed to your chosen federations. No peer or server can decipher the update and no raw data is ever shared. The aggregated global update comes back as a LoRA adapter your runtime attaches at load; your model file is never modified.
- **Efficient local reinforcement learning**: Inference and finetuning run entirely on the user's own machine; raw data never leaves it. QLoRA REINFORCE++ makes on-device RL efficient on consumer GPUs, and the round runs only once the runtime has exited and freed its VRAM.
- **Privacy filter**: Before any gradient is computed, a train-time anonymiser swaps personally identifiable information for consistent surrogates, so personal data can neither be learned nor transmitted.
- **Self-evaluation rewards**: Each conversation is scored by the model's own end-of-session self-evaluation, by how you reacted to its replies, and by verifiable signals read off the tool results in the transcript.
- **Scales to world models**: The same text-native, RL-safe interface extends from agency to world-models, which the federation can train collaboratively and deploy more cheaply than alternative centralized efforts.
- **More than just software**: Federations, continuous model updating, and an exchange of community-trained adapters make Roger a unique ecosystem that improves as more people contribute.

![](assets/divider.PNG)

## Installation & Setup

First, download the repo and navigate to the folder. For Windows users: install under WSL2 to use Flash Attention 2 and get the best performance.

```bash
git clone https://github.com/roger-federated/roger-federated.git # or extract from https://github.com/roger-federated/roger-federated/archive/refs/heads/main.zip
cd roger-federated
```

Note:

- Compatible GPU strongly recommended for the training round's speed. Roger never downloads a model: it trains the very file or cache your runtime served.
- First run writes settings under `~/.roger/config.json`, which includes the federations you participate in. These can be changed at any time.

**Recommended method:**

Additionally requires [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to be installed.

```bash
uv tool install . --torch-backend auto   # installs `roger` globally; auto-picks CUDA/CPU torch
```

**Run:**

Roger wraps the local runtime you already use: prefix its usual command with `roger` and pass the arguments as normal. The user must specify `--port` so that roger can relay it. Therefore, runtimes with conventions that are not OpenAI-compatible (ollama and LM Studio) are not supported yet

```bash
roger vllm serve google/gemma-4-12B-it --port 8000
```
or
```bash
roger llama-server -m model.gguf --port 8080
```

Once the runtime is up, roger tells you whether your federations accept the model you are using, and if not, which models they do accept, so you can switch.

On the first launch of each day, roger pulls your federations' latest model update for the model and keeps it under `~/.roger/federated/`. Every launch then attaches it as a LoRA adapter, so your model file is never modified and no model copy is stored.

Please *exit the runtime properly using Ctrl-C*, so roger can auto-grade the conversation and perform finetune in case enough conversations have accumulated.

<details>
<summary>Remote execution on a trusted machine over SSH</summary>

Roger installs and runs identically on any machine you can SSH into, so a rented GPU instance works exactly like your local one:

- Rent a GPU instance from a provider offering on-demand GPU compute.
- *Important*: Attach block storage that survives instance deletion, in order to persist the state that lives under `~/.roger/`. (Alternatively, you could reverse-tunnel and symlink so that the state is mounted to your local machine.)
- On the remote machine, install the same way, by cloning the repo and running `uv tool install . --torch-backend auto`.
- Optionally run from `tmux` so the session survives an SSH disconnect.

</details>

![](assets/divider.PNG)

<!---
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
-->

## Pricing

- **Individual**: Free. No strings attached.
- **Leechers**: Free for now. Per-pull charges may apply in the future.
- **Enterprise**: Free for now. In the future, pulling model-update may require a license depending on company size.

![](assets/divider.PNG)

## Progress & contributing
The ecosystem is still in development. Below is a non-exhaustive list of to-do items. Of course, we are an open-source community, so **feel free to open an issue or pull request!**

- [x] ~Set up automatic model loading.~
- [x] Write code to handle and record agentic rollouts using HF models.
- [x] ~Give the model agency, i.e., integrate tool use, reasoning, etc., with the LLM as foundation.~
- [x] ~Enable connecting to MCP servers.~
- [x] ~Implement standard tools (e.g., run_command, write_file, prompt_user).~
- [x] ~Replace tool searcher with catalog+load_tools deferred loading (no embedding model needed).~
- [x] ~Implement auto-triggered RAG.~
- [x] ~Allow importing skills and agent.md files.~
- [x] ~Add @path ability.~
- [x] ~Integrate into a CLI application.~
- [x] ~Prettify CLI application.~
- [x] ~Auto read/write memory and create agent folder for temp files.~
- [x] ~Integrate web search and fetch by default.~
- [x] ~Allow resuming conversation after finish.~
- [x] Implement what the finetuning framework considers as rewards.
- [x] LLM-as-judge (self-evaluation inspired by RLSR, SRT, Co-rewarding, meta-evaluation; requires a portion of ground truth).
- [x] Implement automatic QLoRA REINFORCE++.
- [x] Use privacy filter for training data.
- [x] Set up gradient contributing and Bonawitz secure aggregation.
- [x] Build the aggregation server: FedAvg, peer-key distribution, aggregate norm-bounding.
- [x] Use DP without accountant until server is busy, then switch to SMPC.
- [x] Launch the default server as a scale-to-zero container on Scaleway with S3 storage.
- [x] Use signatures and tokens (per-registration secret token proves cohort membership at upload).
- [x] ~Automatic subagent spawning (`spawn_subagent`) with concurrent tool dispatch.~
- [x] ~Native agent loops: `/perpetual` standing tasks with graceful Ctrl-C stop.~
- [x] Wrapper pivot: daily pull of the federation update, attached to llama-server / vllm as a LoRA adapter.
- [x] Wrapper pivot: capture chats off the wire, self-grade them, and train + contribute once the runtime exits.

Deferred:
- [ ] Zero-knowledge integrity proof to verify scale, mod, keys, clip, model fork.
- [ ] Shamir dropout recovery.
- [ ] Always-on mode: always listen, look, and read, and when a keyboard shortcut is entered, automatically infer and continue the user's task using e.g. Touchpoint.
- [ ] Dynalang-style world model for embedded state roll-forward.
- [ ] Centrally coordinated ground-truth gradient injection.
- [ ] Automatic git worktrees for parallel subagent isolation.
