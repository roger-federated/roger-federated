<div align="center">
    <h1>Roger Federated</h1>
    <p><strong>Building the world's first <i>true</i> agent together with the community, using <font color="#818cf8">federated</font> <font color="#4ade80">local</font> <font color="#fb923c">reinforcement learning</font></strong></p>
</div>

---

### Overview
The limitations of the AI transition are coming to light by [far-fetched financial suppositions](https://open.spotify.com/clip/5HzODEnWAegI2Z3NGmu7UV?si=wz0K4G-NSIyTcnPK_1lbBw), [closed models we cannot inspect or steer](https://blog.mozilla.org/en/mozilla/mozilla-open-source-ai-strategy/), generic promises that remain unattained. _Luckily, we are at a point where there's no reason [consumers can't do better with your own hardware, software, and tiny models](https://newsletter.semianalysis.com/p/google-we-have-no-moat-and-neither)._

You and the community can now contribute to the next generation of AI. Not just an LLM with tools, but a purpose-trained agent that is inherently omni-modal. How? By *locally* finetuning a selected *open-source* foundation model on *agentic rollout data*, and subsequently *aggregating* the resulting encrypted model updates (not the data itself) securely with your selected *federations*.

---

### Features
- **Souvereignty**: Inference and finetuning occurs entirely locally on the user's computer (unless otherwise set up), and data is never shared. Only undecipherable gradients are transmitted to selected other users.
- **LoRA finetuning using live reinforcement learning**: Local reinforcement learning finetuning is efficiently performed using LoRA with implicit, automatically detected rewards.
- **Federated learning**: Differentially private gradient are shared with selected specialized federations of other users, without exposing raw training data.
- **Towards omni-modal generality**: A foundation LLM of your choice is finetuned using your federations' gradients, which gradually gives the model inherent agency.
- **Background autonomy**: Headless workers are spawned for asynchronous task completion within isolated, sandboxed virtual environments.

---

### Installation

First:

```bash
git clone git@github.com:thijs-vanweezel/roger-federated.git
cd roger-federated
```

- **Requirements:**
  - Compatible GPU strongly recommended for quantization and speed.
  - First run downloads the selected model (several GB) and writes settings under `~/.roger/config.json`. These can be changed at any time.
  - Linux: `apt install python3-tk` enables the native folder picker; otherwise a text-prompt fallback is used automatically.

- **Recommended:**

Additionally requires [`uv`](https://docs.astral.sh/uv/) to be installed (auto-provisions Python, isolates dependencies).

```bash
uv tool install . --torch-backend auto   # installs `roger` globally; auto-picks CUDA/CPU torch
# or run without installing:
uvx --from . --torch-backend auto roger
```

- **Alternatives:**

These methods additionally require Python 3.10 to be already installed.

Isolated:
```bash
pipx install .
```
Classic venv:
```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

- **GPU notes:**
  - `bitsandbytes` (4/8-bit quantization) is installed automatically only where PyPI ships a wheel: x86-64 Linux and Windows. On other CUDA platforms (e.g. aarch64 Jetson/GH200) install a preview wheel manually, e.g. `pip install --force-reinstall https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_aarch64.whl`.
  - Apple Silicon: GPU (MPS/Metal) is not used yet — only the CUDA path is wired up, so macOS runs unquantized on CPU. PRs adding an MPS check alongside the CUDA check in `src/roger/serving/model_setup.py` are welcome.

- **Run:**

After installing, run `roger` from any terminal. First launch walks you through initial setup. Settings (including the model selection) can subsequently be adjusted in `~/.roger/config.json`.

---

### Use cases
Roger has the potential to perform any digital task. In other words, there is no limit to what you can do with (or delegate to) Roger. Here is a severely non-exhaustive list of examples.

<details>
<summary>Use always-on mode for a 24/7 unsupervised e-marketeer</summary>
...
</details>

<details>
<summary>Connect Touchpoint MCP for assistance in e.g. music production</summary>
...
</details>

<details>
<summary>Automate any MS Office-based task</summary>
...
</details>

<details>
<summary>Fully autonomous development of a social media app</summary>
...
</details>

<details>
<summary>End-to-end Steam game development</summary>
...
</details>

---

### Progress & contributing
This software is still in development. Below is a non-exhaustive list of to-do items. Of course we are an open-source community, so **feel free to open an issue or pull request!**

- [x] Investigate framework and open-source models.
- [x] Set up automatic model loading.
- [x] Write code to handle and record agentic rollouts using HF models and vLLM inference.
- [x] Give the model agency, i.e., integrate tool use, reasoning, etc., with the LLM as foundation.
- [x] Enable connecting to MCP servers.
- [x] Implement standard tools (e.g., run_command, read_file, write_file, search_file, search_dir, prompt_user).
- [x] Replace tool searcher with catalog+load_tools deferred loading (no embedding model needed).
- [x] Implement auto-triggered RAG.
- [x] Allow importing skills and agent.md files.
- [x] Add @path ability.
- [x] Integrate into a CLI application.
- [x] Auto read/write memory and create agent folder for temp files.
- [x] *Congrats, you now have a semi-basic agent.*
- [x] Implement what the finetuning framework considers as rewards.
- [ ] Implement automatic LoRA RL execution.
- [ ] Allow resuming conversation after finish.
- [ ] Implement federated learning-style hives.
- [ ] Sandboxed docker environments.
- [ ] Implement account setup, hives selection, finetuning scheduling, gradient sharing setup, as well as (background/execution) autonomy setup.
- [ ] *Congrats, you now have a shippable.*

Further into the future:
- [ ] Native support for agent loops.
- [ ] Automatic subagent spawning and automatic git worktrees.
- [ ] Accelerate using vLLM.
- [ ] LLM-as-judge.
- [ ] Dynalang-style world model for embedded state roll-forward.
- [ ] Remote control: copy a session code, enter it on our website, continue interacting encrypted through the browser.
- [ ] Always-on mode: always listen, look, and read, and when a keyboard shortcut is entered, automatically infer and continue the user's task using e.g. Touchpoint.