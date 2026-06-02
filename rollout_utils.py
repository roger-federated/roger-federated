import asyncio, torch, transformers, json, inspect
from PIL import Image
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from collections.abc import Callable
from std_tools import get_standard_tools, prompt_user, offer_revert

def make_tool_searcher(tools, model, tokenizer, k=3):
    """Pre-compute tool embeddings; return a search_tools(query) closure for deferred loading."""
    embed_layer = model.get_input_embeddings()
    # Extract name/description from each tool (supports both dict-schema and callable formats)
    tool_info = []
    for t in tools:
        if isinstance(t, dict):
            f = t["function"]
            name, desc = f["name"], f.get("description", "")
        else:
            name, desc = t.__name__, (t.__doc__ or "").split("\n")[0].strip()
        tool_info.append((name, desc, t))
    # Pre-compute avg-pooled token embeddings per tool description (embedding table lookup only)
    tool_embs = {}
    with torch.no_grad():
        for name, desc, _ in tool_info:
            ids = tokenizer.encode(f"{name}: {desc}", add_special_tokens=False, return_tensors="pt").to(embed_layer.weight.device)
            tool_embs[name] = embed_layer(ids).mean(dim=1)

    def tool_searcher(query: str) -> str:
        """Search for a tool by describing the desired functionality.
        Args:
            query: what you want to accomplish (e.g. 'type text into a form field')
        """
        with torch.no_grad():
            ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(embed_layer.weight.device)
            q_emb = embed_layer(ids).mean(dim=1)
        sims = {n: torch.cosine_similarity(q_emb, e).item() for n, e in tool_embs.items()}
        top = sorted(sims, key=sims.get, reverse=True)[:k]
        if sims[top[0]] < 0.5:
            return "No matching tools found. Try rephrasing your query or cancelling the task by not emitting a new tool call."
        from mcp_utils import _strip_schema
        results = []
        for name in top:
            _, _, t = next(x for x in tool_info if x[0] == name)
            results.append(_strip_schema(t) if isinstance(t, dict) else {"name": name, "description": (t.__doc__ or "").strip()})
        return json.dumps(results, indent=2)

    return tool_searcher

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
                        tool_handlers:dict, tool_tokens:tuple[int,int]) -> tuple[dict|None, str|torch.Tensor]:
    """Extract and execute (a single) tool call from output token sequence. If start of tool call is not detected, result is 'terminate'."""
    tokens = sequences.squeeze().tolist()
    # No tool call emitted -> signal termination
    if tool_tokens[0] not in tokens:
        return None, "terminate"
    start_tool_idx = tokens.index(tool_tokens[0])
    end_tool_idx = tokens.index(tool_tokens[1])

    call = json.loads(tokenizer.decode(sequences.squeeze()[start_tool_idx+1:end_tool_idx], skip_special_tokens=False))
    if handler := tool_handlers.get(call["name"], False):
        result = await handler(**call["arguments"]) if inspect.iscoroutinefunction(handler) else handler(**call["arguments"])
    else:
        raise NotImplementedError(f"No tool handler provided for {call['name']}")

    return call, result

async def rollout(model:transformers.modeling_utils.PreTrainedModel, tokenizer:transformers.PreTrainedTokenizer,
                  env, text:str, tool_tokens:tuple[int,int], image:str|Image.Image=None, tools=[],
                  tool_handlers:dict={}, max_steps:int=10, max_new_tokens:int|None=4096, on_token:Callable=None) -> list:
    # Probe once to find the tool-response opener token id (model-agnostic), which is usually the end of a tool call msg
    # TODO: do this also instead of relying on `tool_tokens`?
    _probe = tokenizer.apply_chat_template(
        asst_msg:=[{"role": "assistant", "tool_calls": [{"id": "0", "type": "function", "function": {"name": "_", "arguments": {}}}]}],
        tokenize=True, add_generation_prompt=False
    )["input_ids"]
    str_token_id = _probe[-1]
    
    # Initialize trajectory and provide model with state-getter
    trajectory = []
    # Defer tool loading if too many tools for the prompt
    if len(tools) > 10:
        tool_searcher = make_tool_searcher(tools, model, tokenizer)
        tool_handlers["tool_searcher"] = tool_searcher
        prompt_tools = [tool_searcher]
    else:
        prompt_tools = tools
    # Add standard tools
    std_tools, std_handlers = get_standard_tools()
    prompt_tools += std_tools + [env.get_state]
    tool_handlers = tool_handlers | std_handlers
    tool_handlers[env.get_state.__name__] = env.get_state
    # Create initial input message TODO: allow skills
    messages = [{"role": "system", "content": 
                "\
                You are an agent, will be given a current state and a task description, and must interact with the provided (MCP) tools in order to accomplish the task. \
                After thinking very concisely about intent, provide your tool call in JSON format. A special tool is `get_state`, in which case \
                the overall task status will be provided between a special start state token (i.e., <|state>) and an end state token (i.e., <state|>).\
                "},
                {"role": "user", "content": []}
    ]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    messages[1]["content"] += [{"type": "text", "text": text}]

    # Init and loop conversation
    i = 0
    embed_submodel = model.get_input_embeddings()
    prefix_fn = _make_tool_call_prefix_fn(tokenizer, tool_tokens)
    ple_submodel = next((m for m in model.modules() if hasattr(m, "get_per_layer_inputs")), False)
    state_token_id = tokenizer.encode("<|state>")[0]
    inject_state = False
    past_key_values = None
    # Tokenize the prompt once, then grow input_ids by concatenation
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
        enable_thinking=True, tools=prompt_tools
    ).to(model.device)
    input_ids = inputs["input_ids"]
    while True:
        # Most models are decoder-only, so we can directly pass embeddings
        embeds = embed_submodel(input_ids)
        # Inject state at last <|state> occurence if present
        if inject_state:
            tmp = input_ids[0].tolist()
            state_idx = len(tmp) - tmp[-1::-1].index(state_token_id)
            embeds = torch.cat([embeds[:,:state_idx], state_embeds, embeds[:,state_idx:]], dim=1)

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
            ple = ple_submodel.get_per_layer_inputs(input_ids, None)
            kwargs["per_layer_inputs"] = ple[:, -cached_idx:]
        if mm:=inputs.get("mm_token_type_ids"):
            kwargs["mm_token_type_ids"] = mm[:, -cached_idx:]
        # Generate using fresh streamer
        streamer = transformers.AsyncTextIteratorStreamer(tokenizer, skip_prompt=True)
        loop = asyncio.get_event_loop()
        async def _drain():
            async for text in streamer:
                if on_token: on_token(text)
        generate_task = loop.run_in_executor(None, lambda: model.generate(**kwargs, streamer=streamer))
        outputs, _ = await asyncio.gather(generate_task, _drain())
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
            state_embeds = result
            result = "Task is ongoing. Again, after thinking very concisely about intent based \
                on the provided state, immediately emit your tool call. If no tool call is provided, the task will be assumed done. \n\
                <|state><state|>"
        # Strip the trailing <eos> that prefix_fn forces after <tool_call|> — not part of the format
        gen_ids = outputs.sequences[:, :-1] if outputs.sequences[0, -1] == tokenizer.eos_token_id else outputs.sequences
        # Render tool response delta via template (model-agnostic). asst_msg is there only in case the tool_msg is otherwise ignored
        tool_msg = {"role": "tool", "tool_call_id": "0", "content": str(result)} # TODO: does not support visual results
        delta_full = tokenizer.apply_chat_template(asst_msg + [tool_msg], tokenize=True, add_generation_prompt=True)["input_ids"]
        new_ids = torch.tensor([delta_full[delta_full.index(str_token_id):]], device=model.device, dtype=torch.long)
        input_ids = torch.cat([input_ids, gen_ids, new_ids], dim=1)

        # Check whether we have exceeded max steps
        i += 1
        if i >= max_steps:
            response = prompt_user("Max steps reached. Continue iterating? [y/n]")
            if response.lower() not in ["", "y", "yes"]:
                break

    # Prompt user to revert any files the agent overwrote during the rollout
    offer_revert(prompt_user)

    return trajectory