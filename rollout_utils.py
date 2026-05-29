import torch, transformers, json, inspect
from PIL import Image
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

def _make_tool_call_prefix_fn(tokenizer:transformers.PreTrainedTokenizer, tool_tokens:tuple[int,int]):
    """Restrict generation to valid JSON tool call format while inside a tool call block."""
    # Prepare JSON format
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}
    inner = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))
    def prefix_fn(batch_id, sent_tokens: torch.Tensor):
        tokens = sent_tokens.tolist()
        # Parse start of tool call
        if tool_tokens[0] in tokens and tool_tokens[1] not in tokens:
            idx_start = tokens.index(tool_tokens[0])
            allowed = inner(batch_id, sent_tokens[idx_start + 1:])
            # Tool needs to be closed
            if tokenizer.eos_token_id in allowed: allowed.remove(tokenizer.eos_token_id)
            allowed.append(tool_tokens[1])
            return allowed
        # Force immediate end afterwards
        if tool_tokens[1] in tokens:
            return [tokenizer.eos_token_id]
        return list(range(tokenizer.vocab_size))
    return prefix_fn

async def execute_tools(sequences:torch.Tensor, tokenizer:transformers.PreTrainedTokenizer, 
                        tool_handlers:dict, tool_tokens:tuple[int,int]) -> tuple[str, str|torch.Tensor]:
    """Extract and execute (a single) tool call from output token sequence."""
    start_tool_idx = sequences.squeeze().tolist().index(tool_tokens[0])
    end_tool_idx = sequences.squeeze().tolist().index(tool_tokens[1])

    call = json.loads(tokenizer.decode(sequences.squeeze()[start_tool_idx+1:end_tool_idx], skip_special_tokens=False))
    if handler := tool_handlers.get(call["name"], False):
        result = await handler(**call["arguments"]) if inspect.iscoroutinefunction(handler) else handler(**call["arguments"])
    else:
        raise NotImplementedError(f"No tool handler provided for {call['name']}")

    return call, result

async def rollout(model:transformers.modeling_utils.PreTrainedModel, tokenizer:transformers.PreTrainedTokenizer,
                  env, text:str, tool_tokens:tuple[int,int], image:str|Image.Image=None, tools=[],
                  tool_handlers:dict={}, max_steps:int=10, max_new_tokens:int|None=4096) -> list:
    # Initialize trajectory and provide model with state-getter
    trajectory = []
    tools.append(env.get_state)
    # Create initial input message
    messages = [{"role": "system", "content": 
                """
                You are an agent, will be given a current state and a task description, and must interact with the provided (MCP) tools in order to accomplish the task. \
                After thinking about the long-term and short-term intent, immediately provide your tool call in JSON format. \
                A special tool is `get_state`, in which case the overall task status will be provided between a start state token (i.e., <|state>) and an end state token (i.e., <state|>).
                """},
                {"role": "user", "content": []}
    ]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    messages[1]["content"] += [{"type": "text", "text": text}]

    prefix_fn = _make_tool_call_prefix_fn(tokenizer, tool_tokens)
    ple_submodel = next((m for m in model.modules() if hasattr(m, "get_per_layer_inputs")), False)
    state_token_id = tokenizer.encode("<|state>")[0]
    past_key_values = None
    for i in range(max_steps):
        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
            enable_thinking=True, tools=tools
        ).to(model.device)
        # Remove bos token at subsequent calls TODO: same for eos in cache?
        if i>0 and inputs["input_ids"][0, 0] == tokenizer.bos_token_id:
            inputs["input_ids"] = inputs["input_ids"][:, 1:]
            inputs["attention_mask"] = inputs["attention_mask"][:, 1:]

        # Most models are decoder-only, so we can directly pass (modified) embeddings
        embeds = model.get_input_embeddings()(inputs["input_ids"])
        # Inject current state into prompt IF requested tool call was state query, i.e., result is state
        if i>0 and state_token_id in inputs["input_ids"][0]:
            state_idx = inputs["input_ids"][0].tolist().index(state_token_id)
            embeds = torch.cat([embeds[:,:state_idx], result, embeds[:,state_idx+1:]], dim=1)

        # Attention mask must cover the full sequence: cached prefix + new tokens
        cached_len = past_key_values[0][0].shape[2] if past_key_values else 0
        attn_mask = torch.ones(1, cached_len + embeds.shape[1], device=model.device, dtype=torch.long)

        # Prepare generation arguments
        kwargs = {
            "inputs_embeds": embeds,
            "attention_mask": attn_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_logits": True,
            "return_dict_in_generate": True,
            "prefix_allowed_tokens_fn": prefix_fn, # TODO: this overwrites logits instead of ignoring them, obstructing downstream RL
            "max_new_tokens": max_new_tokens
        }
        # Some models have per-layer embeddings
        if ple_submodel:
            kwargs["mm_token_type_ids"] = inputs.get("mm_token_type_ids")
            kwargs["per_layer_inputs"] = ple_submodel.get_per_layer_inputs(inputs["input_ids"], None)
        # Generate
        outputs = model.generate(**kwargs)
        past_key_values = outputs.past_key_values

        # Parse and execute tool calls from the newly generated tokens only
        generated_ids = outputs.sequences[0]
        print(tokenizer.decode(generated_ids))
        call, result = await execute_tools(generated_ids, tokenizer, tool_handlers, tool_tokens)
        env.update(call, result)

        # Save trajectory
        trajectory.append({"inputs_embeds": embeds, "logits": torch.stack(outputs.logits), "reward": env.get_reward()})
        
        # Check for terminal state
        if env.get_state() == "terminal":
            break

        # Special case if tool call was state query
        if call["name"]==env.get_state.__name__:
            result = "Task ongoing. Again, after thinking about the long-term and short-term intent based \
                on the provided state, immediately emit your tool call. <|state><state|>"
        # Append new message
        messages = [
            {"role": "assistant", "tool_calls": [{"type":"function", "function": call}]},
            {"role": "tool", "content": result}
        ]

    return trajectory