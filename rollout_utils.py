import asyncio, os, sys, torch, transformers, json, inspect, numpy as np, reward_utils
from retrieval import build_index, retrieve, format_context, mark_injected
from PIL import Image
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from collections.abc import Callable
from std_tools import get_standard_tools, prompt_user, offer_revert, maxsteps_checkin
from shell_tools import drain_finished_jobs, pending_jobs, terminate_jobs

def make_tool_loader(tools):
    """Build a terse catalog string + load_tools closure for deferred schema loading.

    Returns (catalog_text, load_tools) where catalog_text lists every tool as
    '- name: one-line description' and load_tools(names) returns full stripped
    schemas for the requested subset.  No embedding model required.
    """
    from mcp_utils import _strip_schema
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


def _build_inner_fn(tokenizer: transformers.PreTrainedTokenizer, tool_names: list[str]):
    """Build constrained-decoding fn with enum-constrained name field; call once per rollout.

    Uses a name-enum schema (prevents misspelled tool names).
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": tool_names},
            "arguments": {"type": "object"}
        },
        "required": ["name", "arguments"]
    }
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))


def _make_prefix_fn(inner, tokenizer: transformers.PreTrainedTokenizer,
                    tool_tokens: tuple[int, int], new_idx: int) -> tuple[Callable, list]:
    """Wrap inner constrained-decoding fn for one generation turn.

    Returns (prefix_fn, mask_log).  mask_log accumulates per-generated-token constraint info:
    None = full vocab (unconstrained step); list = allowed token IDs for that step.
    The REINFORCE++ trainer must renormalize log π(a|s) over the allowed set at each step.

    Balanced open/close counting enables multiple tool calls per turn:
    - opens == closes: free generation (EOS, plain text, or open another call)
    - opens == closes+1: inside a block — constrain to valid JSON, force-close instead of EOS
    """
    mask_log: list = []

    def prefix_fn(batch_id, sent_tokens: torch.Tensor):
        tokens = sent_tokens.squeeze()[new_idx:].tolist()
        opens = tokens.count(tool_tokens[0])
        closes = tokens.count(tool_tokens[1])
        if opens == closes:
            # Not inside any block — free generation
            allowed = list(range(tokenizer.vocab_size))
        else:
            # Inside a block: find last open token, constrain JSON content after it
            last_open = len(tokens) - 1 - tokens[::-1].index(tool_tokens[0])
            allowed = inner(batch_id, sent_tokens[last_open + 1:])
            # Force close token instead of EOS so the block is always well-formed
            if tokenizer.eos_token_id in allowed:
                allowed.remove(tokenizer.eos_token_id)
                allowed.append(tool_tokens[1])
        # None for full vocab avoids storing range(vocab_size) per unconstrained step
        mask_log.append(None if len(allowed) == tokenizer.vocab_size else allowed)
        return allowed

    return prefix_fn, mask_log


async def execute_tools(sequences: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer,
                        tool_handlers: dict,
                        tool_tokens: tuple[int, int]) -> list[tuple[str, any]]:
    """Extract and execute all tool calls from the generated token sequence.

    Returns an ordered list of (name, result) pairs.  Empty list → no call emitted → terminate.
    Errors (malformed JSON, unknown handler, wrong kwargs) become descriptive result strings
    rather than exceptions, so the loop never crashes on a bad call.
    """
    tokens = sequences.tolist()
    results = []
    i = 0
    while i < len(tokens):
        if tokens[i] != tool_tokens[0]:
            i += 1; continue
        # Found open token — locate the matching close
        try:
            end_idx = tokens.index(tool_tokens[1], i + 1)
        except ValueError:
            break  # unterminated block (max_new_tokens cutoff) — stop parsing
        json_tokens = sequences.squeeze()[i + 1:end_idx]
        try:
            call = json.loads(tokenizer.decode(json_tokens, skip_special_tokens=False))
            name = call.get("name", "")
            args = call.get("arguments", {})
        except (json.JSONDecodeError, AttributeError) as e:
            results.append(("__parse_error__", f"Error: malformed tool call — {e}"))
            i = end_idx + 1; continue
        handler = tool_handlers.get(name)
        if not handler:
            result = f"No tool handler provided for '{name}'. Perhaps you misspelled the tool name?"
        else:
            try:
                result = (await handler(**args) if inspect.iscoroutinefunction(handler)
                          else handler(**args))
            except TypeError as e:
                result = f"Error calling '{name}': wrong arguments — {e}"
            except Exception as e:
                result = f"Error calling '{name}': {e}"
        results.append((name, result))
        i = end_idx + 1
    return results


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


async def rollout(model: transformers.modeling_utils.PreTrainedModel,
                  tokenizer: transformers.PreTrainedTokenizer,
                  text: str, tool_tokens: tuple[int, int],
                  image: str | Image.Image = None, tools: list = [],
                  tool_handlers: dict = {}, max_steps: int = 10,
                  max_new_tokens: int | None = 4096,
                  on_token: Callable = None,
                  enable_rag: bool = True, rag_k: int = 3,
                  rag_root: str = None) -> list:

    # Probe once: last token of a dummy assistant-turn-with-tool-calls marks the delta boundary.
    # Used for the max-steps feedback injection path (str_token_id still needed there).
    _probe = tokenizer.apply_chat_template(
        asst_msg := [{"role": "assistant", "tool_calls": [
            {"id": "0", "type": "function", "function": {"name": "_", "arguments": {}}}]}],
        tokenize=True, add_generation_prompt=False
    )["input_ids"]
    str_token_id = _probe[-1]
    # Gen-prompt tokens: every delta is prompt-free; top of while loop prepends exactly one.
    _with_gen = tokenizer.apply_chat_template(asst_msg, tokenize=True, add_generation_prompt=True)["input_ids"]
    gen_prompt = torch.tensor([_with_gen[len(_probe):]], device=model.device, dtype=torch.long)

    trajectory = []

    # Deferred tool loading: when >15 tools, keep only their names in context; full schemas
    # fetched on demand via load_tools(names=[...]).
    if len(tools) > 15:
        catalog_text, load_tools_fn = make_tool_loader(tools)
        tool_handlers["load_tools"] = load_tools_fn
        tools = [load_tools_fn]
    else:
        catalog_text, tools = None, list(tools)
    std_tools_list, std_handlers = get_standard_tools()
    tools += std_tools_list
    tool_handlers = tool_handlers | std_handlers

    # Build constrained-decoding inner fn once for this rollout (schema uses full name list)
    tool_names = [t["function"]["name"] if isinstance(t, dict) else t.__name__ for t in tools]
    inner_fn = _build_inner_fn(tokenizer, tool_names)

    # RAG: build corpus index once (deterministic; no re-indexing on mid-rollout edits)
    rag_index = build_index(rag_root or os.getcwd()) if enable_rag else None
    injected: dict = {}  # path → set[int] of shown 1-indexed line numbers

    # System prompt: environment header + behaviour guidance
    _shell = "PowerShell" if os.name == "nt" else "/bin/sh"
    sys_content = (
        f"Environment: cwd={os.getcwd()}, platform={sys.platform}, shell={_shell}\n"
        "You are an agentic assistant. Given a task, use the provided tools to accomplish it.\n"
        "You may issue multiple tool calls in one turn by emitting them back-to-back in JSON. "
        "Call finish() when the task is complete, or simply stop emitting calls."
    )
    if catalog_text:
        sys_content += (
            "\nAvailable tools (call load_tools(names=[...]) to load a tool's full schema "
            "before using it):\n" + catalog_text
        )
    # RAG start: retrieve on the initial task text, prepend to system prompt
    if rag_index is not None:
        start_hits = retrieve(text, rag_index, rag_k)
        if start_hits:
            sys_content += "\n\n" + format_context(start_hits)
            mark_injected(injected, start_hits)
    messages = [{"role": "system", "content": sys_content},
                {"role": "user", "content": []}]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    messages[1]["content"] += [{"type": "text", "text": text}]

    # Tokenize the prompt once; input_ids grows by concatenation each turn
    i = 0
    past_key_values = None
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True,
        return_tensors="pt", return_dict=True,
        enable_thinking=True, tools=tools
    ).to(model.device)
    input_ids = inputs["input_ids"]

    def result_to_ids(call_results: list[tuple[str, any]]) -> torch.Tensor:
        "Tokenize N (name, result) tool results → token IDs starting from the delta boundary."
        # N-call assistant message so the template renders tool responses in the right format
        dummy_asst = [{"role": "assistant", "tool_calls": [
            {"id": str(k), "type": "function", "function": {"name": name, "arguments": {}}}
            for k, (name, _) in enumerate(call_results)
        ]}]
        tool_msgs = [{"role": "tool", "tool_call_id": str(k), "content": parse_result(res)}
                     for k, (_, res) in enumerate(call_results)]
        ids = tokenizer.apply_chat_template(
            dummy_asst + tool_msgs, tokenize=True, add_generation_prompt=False)["input_ids"]
        return torch.tensor([ids[ids.index(str_token_id):]], device=model.device, dtype=torch.long)

    def bg_msgs():
        """Inject finished background jobs as tool result messages."""
        finished = drain_finished_jobs()
        if not finished:
            return []
        parts = [result_to_ids([("run_command", f"[background] {jid} finished: {out}")])
                 for jid, out in finished]
        return torch.cat(parts, dim=1)

    while True:
        # Prepend generation prompt once per turn; every delta is prompt-free
        input_ids = torch.cat([input_ids, gen_prompt], dim=1)
        new_idx = input_ids.shape[1]
        # Attention mask covers the full sequence; kv cache accounts for the already-processed prefix
        attn_mask = torch.ones(1, input_ids.shape[1], device=model.device, dtype=torch.long)
        prefix_fn, mask_log = _make_prefix_fn(inner_fn, tokenizer, tool_tokens, new_idx)
        # Prepare inputs; model.generate slices the new tokens internally via past_key_values length
        print(tokenizer.decode(input_ids.squeeze(), skip_special_tokens=False))  # debug: see the full prompt each turn
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_logits": True,
            "return_dict_in_generate": True,
            "prefix_allowed_tokens_fn": prefix_fn,
            "max_new_tokens": max_new_tokens
        }
        # Generate with a fresh streamer; _drain() forwards tokens to on_token concurrently
        streamer = transformers.AsyncTextIteratorStreamer(tokenizer, skip_prompt=True)
        loop = asyncio.get_event_loop()
        async def _drain():
            async for chunk in streamer:
                if on_token: on_token(chunk)
        outputs, _ = await asyncio.gather(
            loop.run_in_executor(None, lambda: model.generate(**kwargs, streamer=streamer)),
            _drain()
        )
        gen_ids = (outputs.sequences[:, :-1]
                   if outputs.sequences[0, -1] == tokenizer.eos_token_id
                   else outputs.sequences)
        new_ids = gen_ids.squeeze()[new_idx:].detach()
        past_key_values = outputs.past_key_values

        # Parse and execute all tool calls from the newly generated tokens only
        results = await execute_tools(new_ids, tokenizer, tool_handlers, tool_tokens)

        # Step reward: sum auto_signal over all results in this turn, clamped to [-1, 1]
        step_reward = max(-1.0, min(1.0,
            sum(reward_utils.auto_signal(r) for _, r in results)))
        trajectory.append({
            "gen_token_ids": new_ids,
            "logits": torch.stack(outputs.logits).detach(),
            "masks": mask_log,
            "reward": step_reward,
        })

        if not results:
            # No tool call: wait for any pending background job, then grant another turn
            if pending := pending_jobs():
                await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            else:
                break  # truly done
            bg = bg_msgs()
            input_ids = torch.cat([gen_ids, bg], dim=1) if isinstance(bg, torch.Tensor) else gen_ids
        else:
            # Append tool results; RAG context if relevant; bg jobs — all prompt-free
            tool_ids = result_to_ids(results)
            bg = bg_msgs()
            rag_delta = None
            if rag_index is not None and not any(name == "finish" for name, _ in results):
                asst_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                rag_hits = retrieve(asst_text, rag_index, rag_k, exclude=injected)[:2]
                if rag_hits:
                    rag_ids = tokenizer.apply_chat_template(
                        asst_msg + [{"role": "user", "content": format_context(rag_hits)}],
                        tokenize=True, add_generation_prompt=False
                    )["input_ids"]
                    rag_delta = torch.tensor(
                        [rag_ids[rag_ids.index(str_token_id):]],
                        device=model.device, dtype=torch.long)
                    mark_injected(injected, rag_hits)
            parts = ([gen_ids, tool_ids]
                     + ([rag_delta] if rag_delta is not None else [])
                     + ([bg] if isinstance(bg, torch.Tensor) else []))
            input_ids = torch.cat(parts, dim=1)
            if any(name == "finish" for name, _ in results):
                break

        # Max-steps check-in (interval doubles each time for exponential backoff)
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
                    tokenize=True, add_generation_prompt=False
                )["input_ids"]
                fb_delta = torch.tensor(
                    [fb_ids[fb_ids.index(str_token_id):]], device=model.device, dtype=torch.long)
                input_ids = torch.cat([input_ids, fb_delta], dim=1)
            else: # continue
                trajectory[-1]["reward"] += reward_utils.W_CONTINUE

    terminate_jobs()  # for safety, kill any still-running background commands

    # Terminal reward: revert penalty (if files were changed) or explicit outcome grade
    revert_prompted, n_reverted = offer_revert(prompt_user)
    if not revert_prompted:
        answer = prompt_user("Was the outcome good? [y/n]").strip().lower()
        terminal_r = reward_utils.W_TERMINAL if answer in ("y", "yes") else -reward_utils.W_TERMINAL
    else:
        terminal_r = -n_reverted * reward_utils.W_REVERT
    for entry in trajectory:
        entry["reward"] += terminal_r

    return trajectory
