# TODO: `env` should be a class with `state` and `reward` attributes, and an `update` method that takes in a list of (tool_name, result) tuples and updates the state and reward accordingly.

import torch, transformers, json, inspect
from PIL import Image
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

def _make_tool_call_prefix_fn(tokenizer: transformers.PreTrainedTokenizer, tool_tokens:tuple[int,int]):
    """Restrict generation to valid JSON tool call format while inside a tool call block."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}
    inner = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))
    def prefix_fn(batch_id, sent_tokens: torch.Tensor):
        tokens = sent_tokens.tolist()
        if tool_tokens[0] in tokens and tool_tokens[1] not in tokens:
            # Filter tokens to after and before tool call tokens
            idx_start = tokens.index(tool_tokens[0])
            return inner(batch_id, sent_tokens[idx_start + 1:])
        return list(range(tokenizer.vocab_size))
    return prefix_fn

async def execute_tools(sequences: torch.Tensor, tokenizer, tool_handlers: dict, tool_tokens: tuple[int,int], env) -> list:
    """Extract and execute all tool calls from output token sequence. Returns list of (name, result) tuples."""
    tokens, results, i = sequences.tolist(), [], 0
    while i < len(tokens):
        if tokens[i] == tool_tokens[0]:
            try: j = tokens.index(tool_tokens[1], i + 1)
            except ValueError: break  # unclosed call
            call = json.loads(tokenizer.decode(sequences[i+1:j], skip_special_tokens=False))
            if handler := tool_handlers.get(call["name"]):
                result = await handler(**call["arguments"]) if inspect.iscoroutinefunction(handler) else handler(**call["arguments"])
                results.append((call["name"], result))
            i = j + 1
        else:
            i += 1
    return env.update(results)

async def execute(model:transformers.modeling_utils.PreTrainedModel, tokenizer, env, text:str, tool_tokens:tuple[int,int], image:str|Image.Image=None, tools=[], tool_handlers:dict={}, max_steps=10) -> list:
    # Initialize trajectory and get initial state
    trajectory = []
    # Create initial input message
    messages = [{"role": "system", "content": 
                """
                You are an agent, will be given a current state and a task description, and must interact with the provided (MCP) tools in order to accomplish the task. \
                The current state will be provided between a start state token (i.e., <|state>) and an end state token (i.e., <state|>). \
                After thinking about the long-term and short-term intent, immediately provide your tool call. \
                After having performed this action, a new state will be provided for you to act upon. \
                """},
                {"role": "user", "content": []}
    ]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    messages[1]["content"] += [{"type": "text", "text": text}, {"type": "text", "text": "<|state>"+"<state|>"}]

    prefix_fn = _make_tool_call_prefix_fn(tokenizer, tool_tokens)
    for _ in range(max_steps):
        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=True,
            tools=tools
        ).to(model.device)

        # Most models are decoder-only, so we can directly pass (modified) embeddings
        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        # Some models have per-layer embeddings
        if ple_submodel:=next((m for m in model.modules() if hasattr(m, "get_per_layer_inputs")), False):
            per_layer_inputs = ple_submodel.get_per_layer_inputs(inputs["input_ids"], None)

        # Determine location in embedding for the state representation
        state_token_id = tokenizer.encode("<|state>")[0]
        state_idx = (inputs["input_ids"] == torch.tensor(state_token_id).to(model.device)).reshape(-1)
        # Format current state into prompt
        state_features = env.state
        inputs_embeds[state_idx:state_idx+len(state_features)] = state_features
        
        # Prepare inputs for generation
        kwargs = {
            "attention_mask": inputs["attention_mask"],
            "mm_token_type_ids": inputs["mm_token_type_ids"],
            "use_cache": True,
            "output_logits": True,
            "return_dict_in_generate": True,
            "inputs_embeds": inputs_embeds,
            "prefix_allowed_tokens_fn": prefix_fn
        }
        if ple_submodel:
            kwargs["per_layer_inputs"] = per_layer_inputs
        # Generate next action 
        # TODO: Is this correct since persistent KV caching is required? Or should a custom autoregression be implemented?
        outputs = model.generate(
            **kwargs
        )
        # Parse and execute tool calls; TODO: update env from results
        env = await execute_tools(outputs.sequences[0], tokenizer, tool_handlers, tool_tokens, env)

        # Save trajectory
        trajectory.append({"inputs_embeds": inputs_embeds, "logits": torch.stack(outputs.logits), "reward": env.reward})
        messages = [{"role": "user", "content": [{"type": "text", "text": "Determine your next action based on the new state. <|state>"+"<state|>"}]}]

        if env.state == "terminal":
            break

    return trajectory