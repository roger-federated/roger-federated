This codebase has the following purpose:
- Eventually, it should be an agent such as OpenClaw/OpenHands, enabling MCP action rollouts etc.
- Importantly, the user is allowed to select one of the supported open-source models (imported through HF), such that execution is entirely local. 
- This allows recording the agentic rollouts (or whatever it is called when an agent goes to complete a task begin to end), such that the model can later be locally trained using LoRA RL.
- The generated gradients are sent to my server, and aggregated along with all other user's gradients, and then broadcast back to the users (akin to federated learning).
- The model is used with custom a custom state embedding inserted into the prompt when queried via a special tool call provided by the environment instance.

Over time, the LLM/foundation model becomes finetuned specifically for agentic workflows, which may arguably be better than the current agents which are basically language models with tools.

The codebase should eventually be split into several folders:
- `agency/` -- rollout loop, state encoding, action parsing
- `envs/` -- concrete environment implementations (shell, browser, code)
- `serving/` -- model loading, quantization, inference wrapper
- `training/` -- GRPO trainer, LoRA config, reward functions
- `federated` -- gradient aggregation, differential privacy, strategies

During code generation, be concise and efficient. I.e., implement the minimum changes necessary. Write plentiful information-dense comments.

This code is primarily functional instead of object oriented. There must be a valid reason for statefulness if a class were to be implemented.

Also, account for `.gitignore` in your context.