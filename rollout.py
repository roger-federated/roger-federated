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
        # Remove bos token at subsequent calls
        if i>0 and inputs["input_ids"][0, 0] == tokenizer.bos_token_id:
            inputs["input_ids"] = inputs["input_ids"][:, 1:]
            inputs["attention_mask"] = inputs["attention_mask"][:, 1:]

        # Most models are decoder-only, so we can directly pass (modified) embeddings
        embeds = model.get_input_embeddings()(inputs["input_ids"])
        # Inject current state into prompt
        state_idx = (inputs["input_ids"][0] == state_token_id).nonzero()[-1].item()
        embeds = torch.cat([embeds[...,:state_idx], env.state, embeds[...,state_idx+1:]], dim=1)

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
            "prefix_allowed_tokens_fn": prefix_fn
        }
        # Some models have per-layer embeddings
        if ple_submodel:=next((m for m in model.modules() if hasattr(m, "get_per_layer_inputs")), False):
            kwargs["mm_token_type_ids"] = inputs.get("mm_token_type_ids")
            kwargs["per_layer_inputs"] = ple_submodel.get_per_layer_inputs(inputs["input_ids"], None)
        # Generate
        outputs = model.generate(**kwargs)
        past_key_values = outputs.past_key_values

        # Parse and execute tool calls from the newly generated tokens only
        generated_ids = outputs.sequences[0][embeds.shape[1]:]
        env = await execute_tools(generated_ids, tokenizer, tool_handlers, tool_tokens, env)

        # Save trajectory
        trajectory.append({"inputs_embeds": embeds, "logits": torch.stack(outputs.logits), "reward": env.reward})
        
        # Check for terminal state
        if env.state == "terminal":
            break

        # Append new message
        messages = [
            {"role": "assistant", "content": tokenizer.decode(generated_ids, skip_special_tokens=False)},
            {"role": "user", "content": "Again, after thinking about the long-term and short-term intent based on the new state, immediately determine your tool call. <|state>"+"<state|>"}
        ]

    return trajectory