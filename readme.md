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

### Use cases
<details>
<summary>Connect a desktop interaction MCP for autonomous background tasks</summary>
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
- [ ] Make tool searcher better.
- [ ] Implement auto-triggered RAG.
- [ ] Make it executable.
- [ ] *Congrats, you now have a semi-basic agent.*
- [x] Implement what the finetuning framework considers as rewards.
- [ ] Implement automatic LoRA RL execution.
- [ ] Implement federated learning-style hives.
- [ ] Integrate product into an (CLI) application in which an account is set up, hives are selected, finetuning is scheduled, gradient sharing settings are applied, as well as (background/execution) autonomy settings, and (file) permission settings.
- [ ] *Congrats, you now have a shippable.*

Further into the future:
- [ ] Sandboxed docker environments.
- [ ] LLM-as-judge.
- [ ] Dynalang-style world model for embedded state roll-forward.