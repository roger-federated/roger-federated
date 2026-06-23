<div align="center">
    <h1>Roger Federated</h1>

<strong>Building the world's first <i>true</i> agent together with the community, using $\textcolor{#58a6ff}{\textsf{\textbf{federated}}}$ $\textcolor{#f0883e}{\textsf{\textbf{local}}}$ $\textcolor{#a371f7}{\textsf{\textbf{reinforcement learning}}}$</strong>

</div>

---

### Overview
The limitations of the AI transition are surfacing as a result of [far-fetched financial assumptions](https://open.spotify.com/clip/5HzODEnWAegI2Z3NGmu7UV?si=wz0K4G-NSIyTcnPK_1lbBw), [closed models we cannot inspect nor steer](https://blog.mozilla.org/en/mozilla/mozilla-open-source-ai-strategy/), and generic promises that remain unattained. _Luckily, we are at a crossroads where you, the consumer can do better [with tiny models on your own hardware](https://newsletter.semianalysis.com/p/google-we-have-no-moat-and-neither)._

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

First, download the repo and navigate to the folder:

```bash
git clone git@github.com:thijs-vanweezel/roger-federated.git # or extract from https://github.com/thijs-vanweezel/roger-federated/archive/refs/heads/main.zip
cd roger-federated
```

- **Requirements:**
  - Compatible GPU strongly recommended for quantization and speed.
  - First run downloads the selected model (several GB) and writes settings under `~/.roger/config.json`. These can be changed at any time.
  - Linux: `apt install python3-tk` enables the native folder picker; otherwise a text-prompt fallback is used automatically.

- **Recommended:**

Additionally requires [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to be installed (provisions Python and isolates dependencies).

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

- **MCP servers:**

To give Roger additional functionality, include MCP servers in `~/.roger/mcp.json`. It uses the standard `mcpServers`-format, and the exact schema can therefore be found at your MCP server's provider.

<details>
<summary>Popular servers to get you started</summary>

Drop any of the entries below or more into `~/.roger/mcp.json` and restart Roger. Replace any `<token>`/`<api-key>` placeholder with your own credential, attained from the respective MCP server. Some stdio servers need a one-off install first. Note that the `_comment` field is ignored.

```json
{
  "mcpServers": {
    "github": {
      "_comment": "Repos, issues, PRs, code search; create a fine-grained PAT for the token",
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/",
      "headers": {"Authorization": "Bearer <token>"}
    },
    "context7": {
      "_comment": "Up-to-date, version-correct docs and snippets for any library; free <api-key> at context7.com raises rate limits",
      "type": "http",
      "url": "https://mcp.context7.com/mcp",
      "headers": {"CONTEXT7_API_KEY": "<api-key>"}
    },
    "ms-365": {
      "_comment": "Outlook, Excel, Word, OneDrive, Calendar via Microsoft Graph; opens a browser login on first use",
      "command": "npx",
      "args": ["-y", "@softeria/ms-365-mcp-server"]
    },
    "notion": {
      "_comment": "Search, read and edit your Notion pages and databases; <token> = an internal-integration secret",
      "command": "npx",
      "args": ["-y", "@notionhq/notion-mcp-server"],
      "env": {"NOTION_TOKEN": "<token>"}
    },
    "linear": {
      "_comment": "Create and manage Linear issues, projects and cycles; <token> = a Linear API key",
      "type": "http",
      "url": "https://mcp.linear.app/mcp",
      "headers": {"Authorization": "Bearer <token>"}
    },
    "touchpoint": {
      "_comment": "Interact with your desktop UI. Note: first run `pip install touchpoint-py`",
      "command": "touchpoint-mcp"
    },
    "sentry": {
      "_comment": "Inspect and triage your Sentry errors and issues",
      "type": "http",
      "url": "https://mcp.sentry.dev/mcp",
      "headers": {"Authorization": "Bearer <token>"}
    }
  }
}
```
</details>

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
<summary>Develop e.g. a social media app fully autonomously as a coding agent</summary>
...
</details>

<details>
<summary>Use a native agent loop for end-to-end development of e.g. a Steam game</summary>
...
</details>

---

### Progress & contributing
This software is still in development. Below is a non-exhaustive list of to-do items. Of course, we are an open-source community, so **feel free to open an issue or pull request!**

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
- [ ] Set up (differential) gradient sharing + server/p2p.
- [ ] <ins>Huzzah, we now have a shippable.</ins>

Further into the future:
- [ ] Remote SSH execution.
- [ ] Sandboxed docker environments.
- [ ] Native support for agent loops.
- [ ] Remote control: copy a session code, enter it on our website, continue interacting encrypted through the browser.
- [ ] Always-on mode: always listen, look, and read, and when a keyboard shortcut is entered, automatically infer and continue the user's task using e.g. Touchpoint.
- [ ] Dynalang-style world model for embedded state roll-forward.
- [ ] Implement federated learning-style hives.
- [ ] Automatic subagent spawning and automatic git worktrees.