import asyncio, os, sys, torch, transformers, json, inspect, numpy as np
import roger.training.reward_utils as reward_utils
from roger.agency.retrieval import build_index, retrieve, format_context
from roger.agency.skill_utils import load_instructions, load_memory, discover_skills, make_skill_loader, project_mem_file
from roger.agency.path_utils import expand_at_references, state_dir
from PIL import Image
from lmformatenforcer import JsonSchemaParser
# compat: lm-format-enforcer 0.11.3 imports PreTrainedTokenizerBase from transformers.tokenization_utils, which was removed
import transformers.tokenization_utils as _tu
if not hasattr(_tu, "PreTrainedTokenizerBase"):
    _tu.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from collections.abc import Callable
from roger.tools.std_tools import (get_standard_tools, finish_score, pending_grade, set_user_grade, clear_grade,
                                    pending_backups, apply_revert, maxsteps_checkin)
from roger.tools.shell_tools import drain_finished_jobs, pending_jobs, terminate_jobs, shell_idioms
from roger.serving.model_setup import find_gen_prompt, find_tool_res_id, find_tool_call_tokens, find_think_tokens
from roger.training import recording

def _make_tool_loader(tools):
    """Build a terse catalog string + load_tools closure for deferred schema loading.

    Returns (catalog_text, load_tools) where catalog_text lists every tool as
    '- name: one-line description' and load_tools(names) returns full stripped
    schemas for the requested subset.  No embedding model required.
    """
    from roger.tools.mcp_utils import _strip_schema
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


def _allowed_tokens(inner, tokenizer, tool_tokens, sent_tokens: torch.Tensor, new_idx: int):
    """Returns allowed token IDs at current step, conditioned on sent_tokens, or return None when unconstrained.

    Count open/close tokens to determine constraints:
    - num open tokens == num close tokens: free generation → None (full vocab; nothing built on the hot path)
    - num open tokens == num close tokens + 1: inside a block → constrain to valid JSON, force-close instead of EOS
    """
    tokens = sent_tokens.squeeze()[new_idx:].tolist()
    if tokens.count(tool_tokens[0]) == tokens.count(tool_tokens[1]):
        return None
    # Inside a block: find last open token, constrain JSON content after it
    last_open_idx = len(tokens) - 1 - tokens[::-1].index(tool_tokens[0])
    allowed = list(inner(0, sent_tokens.squeeze()[last_open_idx + 1:]))  # copy: don't mutate enforcer state
    # Force close token instead of EOS so the block is always well-formed
    if tokenizer.eos_token_id in allowed:
        allowed.remove(tokenizer.eos_token_id)
        allowed.append(tool_tokens[1])
    return allowed


def _make_constraint_processor(inner, tokenizer, tool_tokens, new_idx):
    """Logits processor enforcing the tool-call grammar, plus a position-keyed mask log.

    Returns (processor, mask_by_pos). Scores pass through untouched on unconstrained steps (the hot
    path); inside a block they're -inf-masked to the allowed set. The allowed set (or None) is
    recorded into mask_by_pos keyed by the predicted position (input_ids.shape[1]) — keying by
    position, not call order, keeps masks aligned under speculative decoding.
    """
    mask_by_pos: dict = {}

    def processor(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        allowed = _allowed_tokens(inner, tokenizer, tool_tokens, input_ids, new_idx)
        # Record None for near-full-vocab steps too (free JSON-string positions): avoids storing
        # ~vocab-sized lists, and ~full vocab ≈ unconstrained. Enforcement still uses the precise set.
        mask_by_pos[input_ids.shape[1]] = (None if allowed is None or len(allowed) > tokenizer.vocab_size * 0.9
                                           else allowed)
        if allowed is None:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, allowed] = 0.0
        return scores + mask

    return processor, mask_by_pos


def _old_logps(step_logits, masks, token_ids: torch.Tensor, device) -> torch.Tensor:
    """Per-token behaviour log-prob, computed here where the exact allowed-set is known (so we
    store the scalar, not the full [steps, vocab] logits). Re-imposing the decoded mask (None =
    full vocab) makes it identical in definition to the trainer's `new_logp`. 1-D CPU, per token."""
    toks = token_ids.to(device).long()
    out  = torch.empty(toks.numel(), dtype=torch.float32, device=device)
    for t in range(toks.numel()):
        logits  = step_logits[t].squeeze(0).float()         # [vocab]; raw per-step scores
        allowed = masks[t]
        if allowed is not None:                             # restrict to the decoded allowed-set
            m = torch.full_like(logits, float("-inf"))
            m[allowed] = 0.0
            logits = logits + m
        out[t] = torch.log_softmax(logits, dim=-1)[toks[t]]
    return out.detach().cpu()


async def _execute_tools(sequences: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer,
                        tool_handlers: dict,
                        tool_tokens: tuple[int, int],
                        on_tool_call: Callable = None,
                        on_tool_result: Callable = None) -> list[tuple[str, any]]:
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
        if on_tool_call: on_tool_call(name, args)
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
        if on_tool_result: on_tool_result(name, result, args)
        results.append((name, result))
        i = end_idx + 1
    return results


def _parse_result(result):
    """Convert a tool result to an HF content list (list of typed dicts)."""
    if isinstance(result, Image.Image):
        return [{"type": "image", "image": result}]
    if isinstance(result, (list, tuple)):   # mixed/multiple blocks (e.g. MCP image + caption)
        return [blk for item in result for blk in _parse_result(item)]
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


def _result_to_ids(call_results, processor, tool_res_id, device):
    "Tool results → (token-id delta from the <|tool_response> boundary, mm kwargs | None)."
    dummy_asst = [{"role": "assistant", "tool_calls": [
        {"id": str(k), "type": "function", "function": {"name": name, "arguments": {}}}
        for k, (name, _) in enumerate(call_results)]}]
    tool_msgs = [{"role": "tool", "tool_call_id": str(k), "content": _parse_result(res)}
                 for k, (_, res) in enumerate(call_results)]
    # processor.apply_chat_template delegates to the tokenizer for pure text
    enc = processor.apply_chat_template(dummy_asst + tool_msgs, tokenize=True,
        add_generation_prompt=False, return_dict=True, return_tensors="pt").to(device)
    ids = enc.pop("input_ids")
    # Drop the synthetic asst tool-call prefix
    s = (ids[0] == tool_res_id).nonzero()[0].item()
    if all(c["type"] == "text" for m in tool_msgs for c in m["content"]):
        return ids[:, s:], None
    # Slice token-aligned tensors (token_type_ids) to the delta; leave per-image tensors whole.
    full = ids.shape[1]
    mm_extra = {k: (v[:, s:] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == full else v)
             for k, v in enc.items() if k!="attention_mask"}
    return ids[:, s:], mm_extra


def _bg_msgs(processor, tool_res_id, device):
    """Finished background commands, as tool-result token deltas (empty list if none)."""
    finished = drain_finished_jobs()
    if not finished:
        return []
    parts = [_result_to_ids([("run_command", f"[background] {jid} finished: {out}")],
                            processor, tool_res_id, device)[0] for jid, out in finished]
    return torch.cat(parts, dim=1)


def _user_turn_delta(tokenizer, asst_msg, text, device) -> torch.Tensor:
    """Clean user-turn token delta. Suffix-diff (asst_msg vs asst_msg+[user]) avoids the stray
    <|tool_response> that slicing on tool_res_id would wrongly inject before a user turn."""
    base = tokenizer.apply_chat_template(asst_msg, tokenize=True, add_generation_prompt=False)["input_ids"]
    full = tokenizer.apply_chat_template(
        asst_msg + [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=False)["input_ids"]
    return torch.tensor([full[len(base):]], device=device, dtype=torch.long)


def _revert_preamble(backups) -> str:
    """Numbered listing of every file still changed this session + the /revert hint (empty if none)."""
    if not backups:
        return ""
    listing = "\n".join(f"  {k+1}. {orig}" for k, (orig, _b) in enumerate(backups))
    return ("Files changed this session:\n" + listing +
            "\nType '/revert' (or '/revert 1,3') to undo, or just enter your next task.")


async def _await_user_turn(read_turn, trajectory, seg_start) -> str | None:
    """Next user turn (None on Ctrl-D). Non-obstructing revert: list every file still changed this
    session; the user types '/revert'/'/revert 1,3' (penalised -n*W_REVERT over trajectory[seg_start:])
    or just their next task. Backups persist across tasks — a partial revert keeps the rest, so the
    listing accumulates until each file is explicitly reverted."""
    preamble = _revert_preamble(pending_backups())
    while True:
        nxt = await read_turn(preamble)
        preamble = ""                       # show the revert notice only once per offer
        if nxt is None:                     # Ctrl-D → end session
            return None
        nxt = nxt.strip()
        if not nxt:                         # empty line → re-prompt
            continue
        if pending_backups() and nxt.lower().startswith("/revert"):
            n = apply_revert(nxt[len("/revert"):].strip() or "all")
            for entry in trajectory[seg_start:]:
                entry["reward"] -= n * reward_utils.W_REVERT
            preamble = _revert_preamble(pending_backups())   # re-offer any files left
            continue
        return nxt                          # moved on → backups stay pending for next time


async def rollout(model: transformers.modeling_utils.PreTrainedModel,
                  processor,                       # multimodal processor; .tokenizer is the text path
                  text: str,
                  image: str | Image.Image = None, tools: list = [],
                  tool_handlers: dict = {}, max_steps: int = 10,
                  max_new_tokens: int | None = 4096,
                  on_token: Callable = None,
                  on_gen_start: Callable = None,   # (suppress: bool) → None; fired before each generation turn
                  on_gen_end: Callable = None,     # () → None; fired after the turn's tokens are drained
                  on_tool_call: Callable = None,   # (name, args) → None; fired before each call
                  on_tool_result: Callable = None, # (name, result, args) → None; fired after
                  prompt_backend: Callable = None, # replaces input() for all user prompts
                  read_turn: Callable = None,      # async (preamble="") -> next user turn | None (Ctrl-D)
                  root: str = None,                # project root; defaults to os.getcwd()
                  enable_rag: bool = True, rag_k: int = 3,
                  rag_root: str = None,
                  enable_skills: bool = True, skills_root: str = None,
                  enable_memory: bool = True,
                  gen_kwargs: dict = {}) -> list: # extra generate() kwargs (e.g. speculative-decoding: drafter / n-gram)

    # tool_tokens: (open, close) ids bracketing a tool call; probed from the chat template.
    tool_tokens = find_tool_call_tokens(processor.tokenizer)
    # think_tokens: (open, close) reasoning-channel ids, or None; the finish-nudge seeds the open id.
    think_tokens = find_think_tokens(processor.tokenizer)
    # tool_res_id: <|tool_response> boundary; used to slice prompt-free deltas.
    tool_res_id = find_tool_res_id(processor.tokenizer)
    asst_msg = [{"role": "assistant", "tool_calls": [   # reused in feedback injection below
        {"id": "0", "type": "function", "function": {"name": "_", "arguments": {}}}]}]
    # gen_prompt: assistant-turn cue (e.g. <start_of_turn>model\n). MUST be derived from a
    # user-terminated diff — after a tool_call, Gemma-4's template suppresses the cue even with
    # add_generation_prompt=True (the asst_msg diff yields an empty tensor; verified).
    _gp = find_gen_prompt(processor.tokenizer)
    if not _gp:
        import warnings; warnings.warn("gen_prompt is empty — chat template did not append a model-turn cue; model may emit immediate <eos>")
    gen_prompt = torch.tensor([_gp], device=model.device, dtype=torch.long)

    trajectory = []
    first_prompt = text          # session-origin prompt, used as the saved transcript header
    seg_start = 0                # index into trajectory where the current task's steps begin
    run_dir = None               # set on first finish; reused so the session checkpoints one dir

    # Deferred tool loading: when >15 tools, keep only their names in context; full schemas
    # fetched on demand via load_tools(names=[...]).
    if len(tools) > 15:
        catalog_text, load_tools_fn = _make_tool_loader(tools)
        tool_handlers["load_tools"] = load_tools_fn
        tools = [load_tools_fn]
    else:
        catalog_text, tools = None, list(tools)
    std_tools_list, std_handlers = get_standard_tools(prompt_backend=prompt_backend)
    tools += std_tools_list
    tool_handlers = tool_handlers | std_handlers

    # Skills: discover ~/.roger/skills (global) + project .agents/.claude/skills (nested + flat);
    # register load_skill only when skills exist
    _sroot = skills_root or root or os.getcwd()
    skill_catalog, load_skill_fn = None, None
    if enable_skills:
        _skills = discover_skills(_sroot)
        if _skills:
            skill_catalog, load_skill_fn = make_skill_loader(_skills)
            tools.append(load_skill_fn)
            tool_handlers["load_skill"] = load_skill_fn

    # Build constrained-decoding inner fn once for this rollout (schema uses full name list)
    tool_names = [t["function"]["name"] if isinstance(t, dict) else t.__name__ for t in tools]
    inner_fn = _build_inner_fn(processor.tokenizer, tool_names)
    # Forced side-effect-only memory turn: one extra loop iteration after task completion with the
    # name-enum restricted to file ops. Not recorded to trajectory (avoids policy-gradient bias).
    _glob_mem = os.path.join(state_dir(), "memory", "memory.md")
    _proj_mem = project_mem_file(root or os.getcwd())
    _MEM_SEED = (f"Okay.\nLet me very concisely update my memory with what I learned during our entire conversation. "
                 f"User-level facts (e.g., preferences, identity) go in {_glob_mem}; "
                 f"Project-specific facts (e.g., conventions, overview) go in {_proj_mem}.")
    mem_inner  = _build_inner_fn(processor.tokenizer, ["write_file", "edit_file"]) if enable_memory else None
    saving_mem = False
    # Finish-nudge: run one turn seeded with a reasoning preamble; the name-enum stays unconstrained
    _FIN_SEED = ("Now let me decide: did I just have an interaction that I should grade? If it was merely "
                 "a conversation, no further action is needed. Instead, if it was a task where I emitted tool "
                 "calls, I'll call `finish` with an honest self-eval score in the range [-1, 1]. In the latter case, "
                 "I should weigh how directly and efficiently I reached the goal (preferring very few wasted, wrong, "
                 "or redundant steps), and how accurate and complete the result is. 1 for a clean, fully-"
                 "correct solve, around 0 for partial or clumsy, negative if I largely failed.")
    nudging_finish = False
    # Only a task warrants a finish-nudge
    emitted_tool_call = False

    # RAG: build corpus index once (deterministic; no re-indexing on mid-rollout edits)
    rag_index = build_index(rag_root or root or os.getcwd()) if enable_rag else None

    # System prompt: env header + behaviour guidance + shell idioms + instructions + catalogs + RAG
    _shell = "PowerShell" if os.name == "nt" else "/bin/sh"
    sys_content = (
        f"Environment: cwd={os.getcwd()}, platform={sys.platform}, shell={_shell}\n"
        "You are an agentic assistant named 'Roger Federated'. Given a task, use the provided tools to accomplish it.\n"
        "You may issue multiple tool calls in one turn by emitting them back-to-back in JSON. "
        "They will be executed sequentially, unless `background=True. "
        "Call finish(...) whenever you complete an identified task, grading your own work with "
        "an honest self-evaluation score in [-1, 1], i.e., negative if the execution was inaccurate or suboptimal."
        "The user will then give you their next turn.\n\n"
        + shell_idioms()
    )
    # Project instructions: AGENTS.md or CLAUDE.md from cwd (first-found-wins)
    # @path refs in the file are expanded relative to _sroot
    instr = load_instructions(_sroot)
    if instr:
        sys_content += "\n\n" + expand_at_references(instr, _sroot)
    # Persistent memory: always inject the write protocol + current contents (if any)
    if enable_memory:
        sys_content += "\n\n" + load_memory(_sroot)
    if catalog_text:
        sys_content += (
            "\nAvailable tools (call load_tools(names=[...]) to load a tool's full schema "
            "before using it):\n" + catalog_text
        )
    # Skill catalog: just name+description; body fetched on demand via load_skill(name)
    if skill_catalog:
        sys_content += (
            "\nAvailable skills (call load_skill(name) to load instructions before using):\n"
            + skill_catalog
        )
    # RAG start: retrieve on the initial task text, prepend to system prompt
    if rag_index is not None:
        start_hits = retrieve(text, rag_index, rag_k)
        if start_hits:
            sys_content += "\n\n" + format_context(start_hits)
    messages = [{"role": "system", "content": sys_content},
                {"role": "user", "content": []}]
    if image is not None:
        messages[1]["content"] += [{"type": "image", "image": image}]
    # Expand @path refs in the user prompt; RAG query stays on original text
    messages[1]["content"] += [{"type": "text", "text": expand_at_references(text, os.getcwd())}]

    # Tokenize the prompt once; input_ids grows by concatenation each turn. With an image the
    # processor emits the mm tensors; pre-fill them into the cache so the first generate sees them.
    i = 0
    past_key_values = None
    pending_mm = None # mm kwargs
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True,
        return_tensors="pt", return_dict=True,
        enable_thinking=True, tools=tools
    ).to(model.device)
    input_ids = inputs["input_ids"]
    if image is not None:   # feed the image to the first generate (empty cache → its prefill honors it)
        pending_mm = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}

    while True:
        # Prepend generation prompt once per turn; every delta is prompt-free
        input_ids = torch.cat([input_ids, gen_prompt], dim=1)
        if saving_mem:
            # Seed preamble + open token so the turn drives straight into a (restricted) file call
            _seed = processor.tokenizer.encode(_MEM_SEED, add_special_tokens=False) + [tool_tokens[0]]
            input_ids = torch.cat(
                [input_ids, torch.tensor([_seed], device=model.device, dtype=torch.long)], dim=1)
            new_idx = input_ids.shape[1] - 1    # slice from the open token; constraint engages immediately
            active_fn = mem_inner
        elif nudging_finish:
            # Seed the preamble inside the think channel (open id, no tool-open → finish stays optional).
            _seed = ([think_tokens[0]] if think_tokens else []) \
                    + processor.tokenizer.encode(_FIN_SEED, add_special_tokens=False)
            input_ids = torch.cat(
                [input_ids, torch.tensor([_seed], device=model.device, dtype=torch.long)], dim=1)
            new_idx   = input_ids.shape[1]      # generate after the seed; seed is side-effect-only
            active_fn = inner_fn                # unconstrained: finish optional
        else:
            new_idx   = input_ids.shape[1]
            active_fn = inner_fn
        # Attention mask covers the full sequence; kv cache accounts for the already-processed prefix
        attn_mask = torch.ones(1, input_ids.shape[1], device=model.device, dtype=torch.long)
        logits_proc, mask_by_pos = _make_constraint_processor(active_fn, processor.tokenizer, tool_tokens, new_idx)
        # Prepare inputs; model.generate slices the new tokens internally via past_key_values length
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_logits": True,
            "return_dict_in_generate": True,
            "logits_processor": transformers.LogitsProcessorList([logits_proc]),
            "max_new_tokens": max_new_tokens,
            **gen_kwargs,   # assistant_model (drafter) or prompt_lookup_num_tokens (n-gram)
        }
        # Zero-pad token_type_ids to match the length of the appended messages
        if pending_mm is not None:
            mm = dict(pending_mm)
            if "token_type_ids" in mm:
                win = input_ids.shape[1] - (past_key_values.get_seq_length() if past_key_values is not None else 0)
                t = mm["token_type_ids"]
                mm["token_type_ids"] = torch.cat([t, t.new_zeros(1, win - t.shape[1])], dim=1)
            kwargs.update(mm)
        # Generate with a fresh streamer; _drain() forwards tokens to on_token concurrently.
        # suppress=saving_mem: that turn's open delim is seeded into the prompt (never streamed).
        if on_gen_start: on_gen_start(suppress=saving_mem)
        # The nudge's think-open sits in the skipped prompt; feed it so its reasoning collapses.
        if nudging_finish and think_tokens and on_token:
            on_token(processor.tokenizer.decode([think_tokens[0]]))
        streamer = transformers.AsyncTextIteratorStreamer(processor.tokenizer, skip_prompt=True)
        loop = asyncio.get_event_loop()
        async def _drain():
            async for chunk in streamer:
                if on_token: on_token(chunk)
        outputs, _ = await asyncio.gather(
            loop.run_in_executor(None, lambda: model.generate(**kwargs, streamer=streamer)),
            _drain()
        )
        if on_gen_end: on_gen_end()
        gen_ids = (outputs.sequences[:, :-1]
                   if outputs.sequences[0, -1] == processor.tokenizer.eos_token_id
                   else outputs.sequences)
        new_ids = gen_ids.squeeze()[new_idx:].detach().cpu()
        past_key_values = outputs.past_key_values

        # Parse and execute all tool calls from the newly generated tokens only
        results = await _execute_tools(new_ids, processor.tokenizer, tool_handlers, tool_tokens,
                                      on_tool_call=on_tool_call, on_tool_result=on_tool_result)
        if saving_mem:
            break # memory is now written; end
        if results:
            emitted_tool_call = True

        # Nudge turn is side-effect-only: keep its seeded tokens out of the trajectory.
        if not nudging_finish:
            # Step reward: sum auto_signal over all results in this turn, clamped to [-1, 1]
            step_reward = max(-1.0, min(1.0,
                sum(reward_utils.auto_signal(r) for _, r in results)))
            n = int(new_ids.numel())          # new_ids drops a trailing EOS; align everything to n
            # masks keyed by absolute position (not call order: speculative-decoding safe)
            mask_log = [mask_by_pos.get(new_idx + t) for t in range(n)]
            # no gen_token_ids: trainer recovers them as sequence[gen_start:gen_start+len(masks)]
            entry = {
                "gen_start": int(new_idx),
                "old_logp": _old_logps(outputs.logits, mask_log, new_ids, model.device),
                "masks": mask_log,
                "reward": step_reward,
            }
            if pending_mm is not None:   # store the mm tensors for RL replay
                entry["input_mm"] = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in pending_mm.items()}
                pending_mm = None
            trajectory.append(entry)

        # finish checkpoints this task: broadcast its self-eval grade over the task's steps
        # (since the last user turn), then save the running episode.
        if any(name == "finish" for name, _ in results):
            grade = finish_score()
            for entry in trajectory[seg_start:]:
                entry["reward"] += grade
            # Pass the cumulative sequence so the trainer can teacher-force each turn's span;
            # gen_ids grows monotonically across tasks, so it covers every recorded gen_start.
            run_dir = recording.save_run(trajectory, first_prompt, run_dir,
                                         seq_ids=gen_ids.squeeze(0).detach().cpu())
        # Plain answer, no finish() → model thinks it's done. Nudge it once to self-evaluate
        if not results and not (pending := pending_jobs()):
            if not nudging_finish and emitted_tool_call:
                nudging_finish = True
                input_ids = gen_ids
                continue
        # Mutually exclusive: bg-wait (another model turn) | task done (await user) | tool results.
        if not results and pending: 
            # Note: branch not immediately activated when a command is emitted with background=True, because it returns job id
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            bg = _bg_msgs(processor, tool_res_id, model.device)
            input_ids = torch.cat([gen_ids, bg], dim=1) if isinstance(bg, torch.Tensor) else gen_ids
        elif not results or any(name == "finish" for name, _ in results):
            # read_turn's placeholder reminds the user that Ctrl-D quits and saves memory
            next_text = await _await_user_turn(read_turn, trajectory, seg_start) if read_turn else None
            if next_text is None:                 # Ctrl-D → one memory turn (if any), then end
                if enable_memory and not saving_mem:
                    saving_mem = True
                    input_ids = gen_ids
                    continue
                break
            # Inject the user's turn as a prompt-free delta, retaining the KV-cache across tasks.
            input_ids = torch.cat([gen_ids, _user_turn_delta(processor.tokenizer, asst_msg, next_text, model.device)], dim=1)
            seg_start = len(trajectory)           # next finish grades only this fresh segment
            i = 0                                 # fresh max-steps budget per task
            nudging_finish = False                # arm a fresh finish-nudge for the next task
            emitted_tool_call = False             # fresh segment: re-decide task-vs-conversation
            continue
        else:
            tool_ids, pending_mm = _result_to_ids(results, processor, tool_res_id, model.device)  # mm (or None) → next generate
            bg = _bg_msgs(processor, tool_res_id, model.device)
            parts = [gen_ids, tool_ids] + ([bg] if isinstance(bg, torch.Tensor) else [])
            input_ids = torch.cat(parts, dim=1)

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
                fb_delta = _user_turn_delta(processor.tokenizer, asst_msg, feedback_text, model.device)
                input_ids = torch.cat([input_ids, fb_delta], dim=1)
            else: # continue
                trajectory[-1]["reward"] += reward_utils.W_CONTINUE

    terminate_jobs()  # for safety, kill any still-running background commands

    return trajectory
