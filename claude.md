This codebase has the following purpose:
- Eventually, it should be an agent such as OpenClaw/OpenHands, enabling MCP action rollouts etc.
- Importantly, the user is allowed to select one of the supported open-source models (imported through HF), such that execution is entirely local. 
- This allows recording the agentic rollouts (or whatever it is called when an agent goes to complete a task begin to end), such that the model can later be locally trained using LoRA RL.
- The generated gradients are sent to my server, and aggregated along with all other user's gradients, and then broadcast back to the users (akin to federated learning).

Over time, the LLM foundation model becomes finetuned specifically for agentic workflows, which may arguably be better than the current agents which are basically language models with tools.

The `readme.md` contains some potentially unrealistic promises, some of which are not entirely relevant anymore.

During code generation, be concise and space-efficient. Write plentiful dense comments.

Also, account for `.gitignore` in your context.