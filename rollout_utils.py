import asyncio, torch, transformers, json, inspect, numpy as np, reward_utils
from PIL import Image
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from collections.abc import Callable
from std_tools import (get_standard_tools, prompt_user, offer_revert, maxsteps_checkin,
                       drain_finished_jobs, pending_jobs, terminate_jobs)

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

def _make_tool_call_prefix_fn(tokenizer:transformers.PreTrainedTokenizer, tool_tokens:tuple[int,int], new_idx:int=0) -> Callable:
    """Restrict generation to valid JSON tool call format while inside a tool call block."""
    # Prepare JSON format
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]}
    inner = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))
    def prefix_fn(batch_id, sent_tokens: torch.Tensor):
        tokens = sent_tokens.squeeze()[new_idx:].tolist()
        # Parse start of tool call
        if tool_tokens[0] in tokens and tool_tokens[1] not in tokens:
            idx_start = tokens.index(tool_tokens[0])
            allowed = inner(batch_id, sent_tokens[idx_start + 1:])
            # Tool needs to be closed
            if tokenizer.eos_token_id in allowed:
                allowed.remove(tokenizer.eos_token_id)
                allowed.append(tool_tokens[1])
            return allowed
        elif tool_tokens[1] in tokens: # TODO: should we force eos? if not, also account for this in execute_tools
            return [tokenizer.eos_token_id]
        return list(range(tokenizer.vocab_size))
    return prefix_fn

async def execute_tools(sequences:torch.Tensor, tokenizer:transformers.PreTrainedTokenizer,
                        tool_handlers:dict, tool_tokens:tuple[int,int]) -> str|torch.Tensor:
    """Extract and execute (a single) tool call from output token sequence. If start of tool call is not detected, result is 'terminate'."""
    tokens = sequences.squeeze().tolist()
    # No tool call emitted -> signal termination
    if tool_tokens[0] not in tokens:
        return "terminate"
    start_tool_idx = tokens.index(tool_tokens[0])
    end_tool_idx = tokens.index(tool_tokens[1])

    call = json.loads(tokenizer.decode(sequences.squeeze()[start_tool_idx+1:end_tool_idx], skip_special_tokens=False))
    if handler := tool_handlers.get(call["name"], False):
        result = await handler(**call["arguments"]) if inspect.iscoroutinefunction(handler) else handler(**call["arguments"])
    else:
        raise NotImplementedError(f"No tool handler provided for {call['name']}")

    return result

def parse_result(result):
    """Convert a tool result to an HF content list (list of typed dicts)."""
    if isinstance(result, Image.Image):
        return [{"type": "image", "image": result}]
    if isinstance(result, (torch.Tensor, np.ndarray)):
        return [{"type": "audio", "audio": result}]
    return [{"type": "text", "text": str(result)}]

async def rollout(model:transformers.modeling_utils.PreTrainedModel, tokenizer:transformers.PreTrainedTokenizer,
                  text:str, tool_tokens:tuple[int,int], image:str|Image.Image=None, tools=[],
                  tool_handlers:dict={}, max_steps:int=10, max_new_tokens:int|None=4096, on_token:Callable=None) -> list:
    # Probe once to find the tool-response opener token id (model-agnostic)
    _probe = tokenizer.apply_chat_template(
        asst_msg:=[{"role": "assistant", "tool_calls": [{"id": "0", "type": "function", "function": {"name": "_", "arguments": {}}}]}],
        tokenize=True, add_generation_prompt=False
    )["input_ids"]
    str_token_id = _probe[-1]  # last token of the assistant turn; marks the delta boundary

    # Initialize trajectory
    trajectory = []
    # Defer tool loading if too many tools for the prompt
    if len(tools) > 10:
        tool_searcher = make_tool_searcher(tools, model, tokenizer)
        tool_handlers["tool_searcher"] = tool_searcher
        prompt_tools = [tool_searcher]
    else:
        prompt_tools = tools
    std_tools, std_handlers = get_standard_tools()
    prompt_tools += std_tools
    tool_handlers = tool_handlers | std_handlers
    # Create initial input message TODO: allow skills
    messages = [{"role": "system", "content":
                "\
                You are an agent, will be given a current state and a task description, and must interact with the provided (MCP) tools in order to accomplish the task. \
                After thinking very concisely about intent, provide your tool call in JSON format.\
                "},
                {"role": "user", "content": []}
    ]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    messages[1]["content"] += [{"type": "text", "text": text}]

    # Init and loop conversation
    i = 0
    past_key_values = None
    # Tokenize the prompt once, then grow input_ids by concatenation
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
        enable_thinking=True, tools=prompt_tools
    ).to(model.device)
    input_ids = inputs["input_ids"]
    new_idx = input_ids.shape[1]
    # Finished background commands, surfacing without polling
    # asst_msg prefixes the delta so the template renders it; sliced off via str_token_id.
    def result_to_ids(tool_result):
        tool_msg = [{"role": "tool", "content": tool_result}]
        tool_ids = tokenizer.apply_chat_template(asst_msg + tool_msg, tokenize=True, add_generation_prompt=True)["input_ids"]
        return torch.tensor([tool_ids[tool_ids.index(str_token_id):]], device=model.device, dtype=torch.long)
    bg_msgs = lambda: torch.cat(
        [result_to_ids(f"[background] {jid} finished: {str(out)}") for jid, out in finished_jobs], dim=1
    ) if (finished_jobs:=drain_finished_jobs()) else []

    while True:
        # Attention mask covers the full sequence (cached prefix accounted for by past_key_values)
        attn_mask = torch.ones(1, input_ids.shape[1], device=model.device, dtype=torch.long)

        # Prepare inputs and rely on internal slicing based on kv cache length
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_logits": True,
            "return_dict_in_generate": True,
            "prefix_allowed_tokens_fn": _make_tool_call_prefix_fn(tokenizer, tool_tokens, new_idx), # TODO: this overwrites logits instead of ignoring them, obstructing downstream RL
            "max_new_tokens": max_new_tokens
        }
        # Generate using fresh streamer
        streamer = transformers.AsyncTextIteratorStreamer(tokenizer, skip_prompt=True)
        loop = asyncio.get_event_loop()
        async def _drain():
            async for text in streamer:
                if on_token: on_token(text)
        generate_task = loop.run_in_executor(None, lambda: model.generate(**kwargs, streamer=streamer))
        outputs, _ = await asyncio.gather(generate_task, _drain())
        gen_ids = outputs.sequences[:, :-1] if outputs.sequences[0, -1] == tokenizer.eos_token_id else outputs.sequences
        new_ids = gen_ids.squeeze()[:, new_idx:].detach()
        past_key_values = outputs.past_key_values

        # Parse and execute tool calls from the newly generated tokens only
        result = await execute_tools(new_ids, tokenizer, tool_handlers, tool_tokens)

        trajectory.append({
            "gen_token_ids": new_ids,
            "logits": torch.stack(outputs.logits).detach(),
            "reward": reward_utils.auto_signal(result),
        })

        # Model wants to stop: wait for a still-running command, give another turn
        if result == "terminate":
            if pending := pending_jobs():
                await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            else:
                break
            input_ids = torch.cat([gen_ids, bg_msgs()], dim=1)
        else:
            tool_ids = result_to_ids(parse_result(result))
            input_ids = torch.cat([gen_ids, tool_ids, bg_msgs()], dim=1)
        new_idx = input_ids.shape[1]

        # Check whether we have exceeded max steps
        i += 1
        if i >= max_steps:
            action, feedback_text = maxsteps_checkin()
            if action == "abort":
                trajectory[-1]["reward"] -= reward_utils.W_ABORT
                break
            max_steps += max_steps
            if action == "feedback":
                trajectory[-1]["reward"] -= reward_utils.W_FEEDBACK
                fb_ids = tokenizer.apply_chat_template(
                    asst_msg + [{"role": "user", "content": feedback_text}],
                    tokenize=True, add_generation_prompt=True
                )["input_ids"]
                fb_delta = torch.tensor([fb_ids[fb_ids.index(str_token_id):]], device=model.device, dtype=torch.long)
                input_ids = torch.cat([input_ids, fb_delta], dim=1)
                new_idx = input_ids.shape[1]
            else:  # "continue"
                trajectory[-1]["reward"] += reward_utils.W_CONTINUE

    terminate_jobs()  # for safety, kill any background commands still running at episode end

    # Broadcast terminal reward (revert penalty or explicit outcome grade) to all steps
    revert_prompted, n_reverted = offer_revert(prompt_user)
    if not revert_prompted:
        answer = prompt_user("Was the outcome good? [y/n]").strip().lower()
        terminal_r = reward_utils.W_TERMINAL if answer in ("y", "yes") else -reward_utils.W_TERMINAL
    else:
        terminal_r = -n_reverted * reward_utils.W_REVERT
    for entry in trajectory:
        entry["reward"] += terminal_r

    return trajectory
