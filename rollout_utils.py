import asyncio, torch, transformers, json, inspect, numpy as np, reward_utils
from PIL import Image
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from collections.abc import Callable
from std_tools import (get_standard_tools, prompt_user, offer_revert, maxsteps_checkin,
                       drain_finished_jobs, pending_jobs, terminate_jobs)

def make_tool_loader(tools):
    """Build a terse catalog string + load_tools closure for deferred schema loading.

    Returns (catalog_text, load_tools) where catalog_text lists every tool as
    '- name: one-line description' and load_tools(names) returns full stripped
    schemas for the requested subset.  No embedding model required.
    """
    from mcp_utils import _strip_schema
    # Build index: name -> (one-liner, original tool object)
    index = {}
    for t in tools:
        if isinstance(t, dict):
            f = t["function"]
            name = f["name"]
            desc = f.get("description", "").split("\n")[0].strip()
        else:
            name = t.__name__
            desc = (t.__doc__ or "").split("\n")[0].strip()
        index[name] = (desc, t)

    catalog_text = "\n".join(f"- {n}: {d}" for n, (d, _) in index.items())

    def load_tools(names: list) -> str:
        """Load full schemas for named tools so you can call them.
        Args:
            names: list of tool names to load (from the catalog)
        Returns: JSON array of tool schemas
        """
        results = []
        for n in names:
            if n not in index:
                results.append({"name": n, "error": "unknown tool name"})
                continue
            _, t = index[n]
            results.append(_strip_schema(t) if isinstance(t, dict) else
                           {"name": n, "description": (t.__doc__ or "").strip()})
        return json.dumps(results, indent=2)

    return catalog_text, load_tools

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
    tokens = sequences.tolist()
    # No tool call emitted -> signal termination
    if tool_tokens[0] not in tokens:
        return "terminate"
    start_tool_idx = tokens.index(tool_tokens[0])
    end_tool_idx = tokens.index(tool_tokens[1])

    call = json.loads(tokenizer.decode(sequences.squeeze()[start_tool_idx+1:end_tool_idx], skip_special_tokens=False))
    if handler := tool_handlers.get(call["name"], False):
        result = await handler(**call["arguments"]) if inspect.iscoroutinefunction(handler) else handler(**call["arguments"])
    else:
        result = f"No tool handler provided for {call['name']}. Perhaps you misspelled the tool name?"

    return result

def parse_result(result):
    """Convert a tool result to an HF content list (list of typed dicts)."""
    if isinstance(result, Image.Image):
        return [{"type": "image", "image": result}]
    if isinstance(result, (torch.Tensor, np.ndarray)):  # torchaudio / librosa
        return [{"type": "audio", "audio": result}]
    try:
        from pydub import AudioSegment
        if isinstance(result, AudioSegment):
            arr = np.array(result.get_array_of_samples(), dtype=np.float32)
            arr /= float(2 ** (result.sample_width * 8 - 1))  # normalise to [-1, 1]
            return [{"type": "audio", "audio": arr}]
    except ImportError:
        pass
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
    # Defer tool loading if too many tools: keep terse name catalog in context; model calls
    # load_tools(names=[...]) to pull full schemas before using them.
    if len(tools) > 15:
        catalog_text, load_tools_fn = make_tool_loader(tools)
        tool_handlers["load_tools"] = load_tools_fn
        tools = [load_tools_fn]
    else:
        catalog_text, tools = None, list(tools)
    std_tools, std_handlers = get_standard_tools()
    tools += std_tools
    tool_handlers = tool_handlers | std_handlers
    # Create initial input message
    sys_content = (
        "You are an agent, will be given a current state and a task description, and must "
        "interact with the provided (MCP) tools in order to accomplish the task. "
        "After thinking very concisely about intent, provide your tool call in JSON format."
    )
    if catalog_text:
        sys_content += (
            "\nAvailable tools (call load_tools(names=[...]) to load a tool's full schema "
            "before using it):\n" + catalog_text
        )
    messages = [{"role": "system", "content": sys_content},
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
        enable_thinking=True, tools=tools
    ).to(model.device)
    input_ids = inputs["input_ids"]
    new_idx = input_ids.shape[1]
    # Get tool result token ids (asst_msg prefixes the delta so the template renders it; sliced off via str_token_id)
    def result_to_ids(tool_result):
        tool_msg = [{"role": "tool", "content": tool_result}]
        tool_ids = tokenizer.apply_chat_template(asst_msg + tool_msg, tokenize=True, add_generation_prompt=True)["input_ids"]
        return torch.tensor([tool_ids[tool_ids.index(str_token_id):]], device=model.device, dtype=torch.long)
    # Finished background commands, surfacing without polling
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
        new_ids = gen_ids.squeeze()[new_idx:].detach()
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
            bg = bg_msgs()
            input_ids = torch.cat([gen_ids, bg], dim=1) if isinstance(bg, torch.Tensor) else gen_ids
        else:
            tool_ids = result_to_ids(parse_result(result))
            bg = bg_msgs()
            parts = [gen_ids, tool_ids] + ([bg] if isinstance(bg, torch.Tensor) else [])
            input_ids = torch.cat(parts, dim=1)
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
