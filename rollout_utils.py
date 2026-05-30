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
            if tokenizer.eos_token_id in allowed: 
                allowed.remove(tokenizer.eos_token_id)
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
                After concisely thinking about the long-term and short-term intent, immediately provide your tool call in JSON format. \
                A special tool is `get_state`, in which case the overall task status will be provided between a start state token (i.e., <|state>) and an end state token (i.e., <state|>).
                """},
                {"role": "user", "content": []}
    ]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    messages[1]["content"] += [{"type": "text", "text": text}]

    # Init and loop conversation
    embed_submodel = model.get_input_embeddings()
    prefix_fn = _make_tool_call_prefix_fn(tokenizer, tool_tokens)
    ple_submodel = next((m for m in model.modules() if hasattr(m, "get_per_layer_inputs")), False)
    state_token_id = tokenizer.encode("<|state>")[0]
    inject_state = False
    past_key_values = None
    for _ in range(max_steps):
        # Tokenize full conversation every step, internally sliced when kv cache is passed
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
            enable_thinking=True, tools=tools
        ).to(model.device)

        # Most models are decoder-only, so we can directly pass embeddings
        embeds = embed_submodel(inputs["input_ids"])
        # Inject state at last <|state> occurence if present
        if inject_state:
            tmp = inputs["input_ids"][0].tolist()
            state_idx = len(tmp) - tmp[-1::-1].index(state_token_id)
            embeds = torch.cat([embeds[:,:state_idx], result, embeds[:,state_idx:]], dim=1)

        # Attention mask covers the full sequence (cached prefix accounted for by past_key_values)
        attn_mask = torch.ones(1, embeds.shape[1], device=model.device, dtype=torch.long)

        # Prepare inputs and rely on internal slicing based on kv cache length
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
        # Keep track of what is cached, i.e., all but the newly inserted message
        if past_key_values: cached_idx = embeds.shape[1] - past_key_values.get_seq_length()
        else: cached_idx = 0
        # Some models use per-layer-embeddings
        if ple_submodel:
            ple = ple_submodel.get_per_layer_inputs(inputs["input_ids"], None)
            kwargs["per_layer_inputs"] = ple[:, -cached_idx:]
        if mm:=inputs.get("mm_token_type_ids"):
            kwargs["mm_token_type_ids"] = mm[:, -cached_idx:]
        # Generate
        outputs = model.generate(**kwargs)
        generated_ids = outputs.sequences[0]
        past_key_values = outputs.past_key_values

        # Parse and execute tool calls from the newly generated tokens only
        call, result = await execute_tools(generated_ids, tokenizer, tool_handlers, tool_tokens)
        env.update(call, result)

        # Save trajectory
        trajectory.append({"inputs_embeds": embeds, "logits": torch.stack(outputs.logits), "reward": env.get_reward()})

        # Check for terminal state
        if env.get_state() == "terminal":
            break

        # Special case if tool call was state query
        if inject_state:=(call["name"]==env.get_state.__name__):
            result = "Task ongoing. Again, after concisely thinking about the long-term and short-term intent based \
                on the provided state, immediately emit your tool call. <|state><state|>"
        # Accumulate conversation
        messages += [
            {"role": "assistant", "content": tokenizer.decode(generated_ids, skip_special_tokens=False)},
            {"role": "tool", "content": result}
        ]

    return trajectory