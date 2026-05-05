# <div style="text-align: center;">Project name pending</div>

### <div style="text-align: center;"> An OS-level agent that uses real-time reinforcement learning to learn from you and the rest of the community. </div>

---
### Overview
**Under the reign of big tech companies, your decisional capacity is going to be put in the hands of models you cannot inspect nor steer.** But fear not; you and the community can now contribute to the next generation of AI. Not just an LLM with tools, but a purpose-trained agent that is inherently omni-modal. How? By *locally* finetuning a selected open-source foundation model on *OS-level data* generated during your computer usage, and *sharing* the resulting encrypted model updates (not the data itself) securely with your selected *federations*.

---

### Features
- **Souvereignty**: Inference and finetuning occurs entirely locally on the user's computer (unless otherwise set up), and data is never shared. Only undecipherable gradients are transmitted to selected other users.
- **LoRA RL-based finetuning**: Local reinforcement learning finetuning is efficiently performed using LoRA and implicit, automatically detected rewards.
- **Federated learning**: Differentially private gradient are shared with selected specialized federations of other users, without exposing raw training data.
- **Towards omni-modal generality**: A foundation LLM of your choice is finetuned using your federations' gradients, which gradually gives the model inherent agency.
- **Background autonomy**: Headless workers are spawned for asynchronous task completion within isolated, sandboxed virtual environments.
- **Verifiable security**: A cryptographic ledger of proof of contribution ensures the integrity and quality of model updates.

---

### Use cases
...

---

### Progress & contributing
The product is still in development. Below is a non-exhaustive list of to-do items. Of course we are an open-source community, so **feel free to open an issue or pull request!**

- [x] Open a git repository and describe the project details in an md file.
- [x] ~~Investigate universal model formats (e.g., ONNX) and open-source model implementations (e.g., Gemma-4), and whether the parameters are mutable for finetuning purposes.~~
- [ ] ~~Lay out a finetuning framework.~~
- [ ] ~~Make the finetuning framework compatible with LoRA.~~
- [ ] Write code to handle and record agentic rollouts using HF models and vLLM inference.
- [ ] Investigate how to programmatically interact with the OS.
- [ ] Give the model agency, i.e., integrate tool use, reasoning, etc., with the LLM as foundation.
- [ ] Give the agent autonomy, i.e., ability to execute in a background VM/sandbox.
- [ ] Implement what the finetuning framework considers as rewards.
- [ ] Congrats, you now have a working prototype.
- [ ] Implement federated learning-style hives.
- [ ] Integrate product into an application in which an account is set up, hives are selected, finetuning is scheduled, gradient sharing settings are applied, as well as (background/execution) autonomy settings, and (file) permission settings.
- [ ] Congrats, you now have a product.