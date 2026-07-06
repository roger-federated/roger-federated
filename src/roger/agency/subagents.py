"""Sub-agent spawning via turn-synchronized batched decoding.

A sub-agent is another conversation over the *same* in-VRAM model — no reload, no subprocess.
The main rollout (agency/rollout_utils.rollout) keeps its cache-reuse fast path unchanged; when it
spawns sub-agents it parks and drives a SubAgentScheduler that advances all live sub-agents in
lockstep: one batched model.generate() per turn over every *ready* sub-agent (reprefill — each
turn left-pads the agents' full sequences into one batch and builds a fresh cache; see
scratchpad prototype, this avoids all sliding-window KV surgery), then each sub-agent's tools run
concurrently, then the next batched turn. A sub-agent that is mid-tool sits out the batch and
rejoins when ready ("detrimental blockers" don't stall the others).

Design decisions (see plan i-think-it-s-time-jolly-sifakis + memory subagent-spawning-design):
  * one-shot fire-and-collect: spawn_subagent(task) returns a handle immediately; the parent gets
    the child's final answer via the existing background-job wait/inject seam.
  * every agent gets its own ToolSession (isolated jobs/backups/grade) and the full toolset + MCP.
  * recursion allowed, bounded by a shared semaphore over concurrently-alive sub-agents.
  * sub-agents are RL-recorded like the main agent (own trajectory + run_dir + self-grade).
"""
import asyncio, os, sys, torch, transformers
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable

import roger.training.reward_utils as reward_utils
from roger.tools.session import ToolSession
from roger.tools.std_tools import get_standard_tools
from roger.tools.shell_tools import shell_idioms
from roger.agency.skill_utils import load_instructions, discover_skills, make_skill_loader
from roger.agency.path_utils import expand_at_references
from roger.training import recording


# The batched core + per-slot step logic live here; grammar/dispatch helpers are reused from
# rollout_utils (imported lazily inside functions to avoid an import cycle: rollout_utils imports
# this module to wire the spawn tool).

SUMMARY_SUFFIX = "\n\nAfter completing this request, summarize your findings."


@dataclass
class SubAgent:
    """One sub-agent slot: its full token sequence, tool state, grammar, trajectory, and lifecycle."""
    task: str
    handle: str
    session: ToolSession
    tools: list
    tool_handlers: dict
    inner_fn: Callable                    # constrained-decoding grammar (name-enum for its toolset)
    input_ids: torch.Tensor = None        # full sequence so far [1, L]; reprefilled each batched turn
    trajectory: list = field(default_factory=list)
    run_dir: str = None
    seg_start: int = 0
    first_prompt: str = ""
    emitted_tool_call: bool = False
    parent: "SubAgent" = None             # the slot that spawned this one; None ⇒ spawned by the main agent
    phase: str = "ready"                  # "ready" (will generate this turn) | "parked" (awaiting own children)
    steps: int = 0
    done: bool = False
    collected: bool = False               # result already delivered to parent/main
    result: str = ""                      # final answer handed back to the parent
    # per-turn scratch (filled by batched_generate, consumed by after_generate)
    mask_by_pos: dict = field(default_factory=dict)
    new_ids: torch.Tensor = None
    step_logits: list = field(default_factory=list)
    masks: list = field(default_factory=list)


def _stop_ids(model) -> set:
    """The token ids that end a turn (eos + any end_of_turn), from generation_config."""
    e = model.generation_config.eos_token_id
    ids = set(e) if isinstance(e, (list, tuple)) else {e}
    ids.discard(None)
    return ids


def _cut_at_stop(gen: list[int], stop: set) -> list[int]:
    """Trim a row's generated tokens at the first stop token (exclusive), matching the single-agent
    loop's 'drop the trailing eos' behaviour. Rows that finished early are pad-filled by generate."""
    for i, t in enumerate(gen):
        if t in stop:
            return gen[:i]
    return gen


def batched_generate(slots: list[SubAgent], model, processor, tool_tokens, max_new_tokens: int):
    """Advance every slot by one turn in a single batched forward (reprefill).

    Left-pads each slot's full sequence into one batch, runs one constrained generate with a
    per-row grammar + per-row mask capture, then splits the result back onto each slot
    (slot.new_ids / step_logits / masks). No KV cache is persisted across turns.
    """
    from roger.agency.rollout_utils import _allowed_tokens   # lazy: avoid import cycle
    tok = processor.tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop = _stop_ids(model)
    Smax = max(s.input_ids.shape[1] for s in slots)
    dev = model.device

    rows, attn = [], []
    for s in slots:
        n = s.input_ids.shape[1]; padlen = Smax - n
        rows.append(torch.cat([torch.full((1, padlen), pad, device=dev, dtype=torch.long), s.input_ids], dim=1))
        attn.append(torch.cat([torch.zeros(1, padlen, device=dev, dtype=torch.long),
                               torch.ones(1, n, device=dev, dtype=torch.long)], dim=1))
        s.mask_by_pos = {}
    batch_ids = torch.cat(rows, 0); attn = torch.cat(attn, 0)
    V = tok.vocab_size

    def processor_fn(input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, S] (grows each step); scores: [B, V]. Constrain each row by its own grammar.
        pos = input_ids.shape[1]
        for b, s in enumerate(slots):
            allowed = _allowed_tokens(s.inner_fn, tok, tool_tokens, input_ids[b:b+1], Smax)
            s.mask_by_pos[pos] = (None if allowed is None or len(allowed) > V * 0.9 else allowed)
            if allowed is not None:
                m = torch.full_like(scores[b], float("-inf"))
                m[allowed] = 0.0
                scores[b] = scores[b] + m
        return scores

    out = model.generate(
        batch_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
        use_cache=True, output_logits=True, return_dict_in_generate=True,
        logits_processor=transformers.LogitsProcessorList([processor_fn]),
    )
    for b, s in enumerate(slots):
        gen = out.sequences[b, Smax:].tolist()
        new = _cut_at_stop(gen, stop)
        n = len(new)
        s.new_ids = torch.tensor(new, dtype=torch.long)
        s.step_logits = [out.logits[t][b:b+1] for t in range(n)]   # [1, V] per generated step
        s.masks = [s.mask_by_pos.get(Smax + t) for t in range(n)]


async def _grade_and_record(slot: SubAgent, seg_end: int, model, processor, tool_tokens, root):
    """Self-grade the finished segment, fold the grade into its rewards, and checkpoint the run."""
    from roger.agency.rollout_utils import _build_inner_fn, _allowed_tokens, _execute_tools, _GRADE_SEED
    tok = processor.tokenizer
    if slot.emitted_tool_call:
        grade_inner = _build_inner_fn(tok, ["_grade"])
        seed = tok.encode(_GRADE_SEED, add_special_tokens=False) + [tool_tokens[0]]
        ids = torch.cat([slot.input_ids, torch.tensor([seed], device=model.device, dtype=torch.long)], dim=1)
        base = ids.shape[1]

        def proc(input_ids, scores):
            allowed = _allowed_tokens(grade_inner, tok, tool_tokens, input_ids, base)
            if allowed is not None:
                m = torch.full_like(scores[0], float("-inf")); m[allowed] = 0.0
                scores[0] = scores[0] + m
            return scores

        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
                             use_cache=True, return_dict_in_generate=True,
                             logits_processor=transformers.LogitsProcessorList([proc]))
        slot.session.clear_grade()
        await _execute_tools(out.sequences.squeeze()[base:], tok,
                             {"_grade": slot.session.record_grade}, tool_tokens)
        grade = slot.session.grade_value() or 0.0
        for e in slot.trajectory[slot.seg_start:seg_end]:
            e["reward"] += grade
    # Persist the sub-agent run as its own RL episode (own run_dir; the trainer scans all runs).
    if slot.trajectory:
        slot.run_dir = recording.save_run(slot.trajectory, slot.first_prompt, slot.run_dir,
                                          seq_ids=slot.input_ids.squeeze(0).detach().cpu())


def build_subagent_prompt(root: str) -> str:
    """Lean system prompt for a sub-agent: env header + guidance + shell idioms + project instructions.
    Deliberately omits RAG/memory (the parent handles those); skills are attached via the tool list."""
    _shell = "PowerShell" if os.name == "nt" else "/bin/sh"
    sys_content = (
        f"cwd: {os.getcwd()}, platform: {sys.platform}, shell: {_shell}, "
        f"date/time: {datetime.now(timezone.utc).isoformat()}\n"
        "You are a sub-agent spawned to accomplish one focused task. Use the provided tools to do it, "
        "then stop emitting tool calls and give your final answer.\n"
        "You may issue multiple tool calls in one turn by emitting them back-to-back in JSON. "
        "They run sequentially unless background=True.\n"
        "Prefer empirical discovery over recalling from your internal knowledge.\n\n"
        + shell_idioms()
    )
    instr = load_instructions(root)
    if instr:
        sys_content += "\n\n" + expand_at_references(instr, root)
    return sys_content


class SubAgentScheduler:
    """Drives all live sub-agents in lockstep via turn-synchronized batched decoding.

    The main agent (agency/rollout_utils.rollout) owns an instance and calls run_until_progress()
    while parked awaiting its sub-agents. Each call advances every *ready* sub-agent one batched turn,
    runs their tools concurrently, injects finished sub-agents' results back into their parents
    (recursion) or returns them to the main agent, and repeats until a main-parented sub-agent
    finishes (progress) or nothing is left to run. The concurrently-alive count is capped by max_alive.
    """

    def __init__(self, model, processor, *, max_alive: int, max_steps: int, root: str,
                 max_new_tokens: int = 4096, mcp_tools=(), mcp_handlers=None, prompt_backend=None,
                 enable_skills: bool = True, skills_root: str = None,
                 policy_file: str = "command_policy.txt",
                 on_tool_call: Callable = None, on_tool_result: Callable = None):
        from roger.loading.model_setup import find_tool_call_tokens, find_tool_res_id, find_gen_prompt
        self.model = model; self.processor = processor; self.tok = processor.tokenizer
        self.tool_tokens = find_tool_call_tokens(self.tok)
        self.tool_res_id = find_tool_res_id(self.tok)
        self.gen_prompt = torch.tensor([find_gen_prompt(self.tok)], device=model.device, dtype=torch.long)
        self.max_alive = max_alive; self.max_steps = max_steps; self.root = root
        self.max_new_tokens = max_new_tokens
        self.mcp_tools = list(mcp_tools); self.mcp_handlers = dict(mcp_handlers or {})
        self.prompt_backend = prompt_backend
        self.enable_skills = enable_skills; self.skills_root = skills_root or root
        self.policy_file = policy_file
        self.on_tool_call = on_tool_call; self.on_tool_result = on_tool_result
        self.slots: list[SubAgent] = []
        self._seq = 0
        self.to_main: list[tuple[str, str]] = []   # finished (handle, result) whose parent is the main agent

    # -- lifecycle --------------------------------------------------------------------------------
    def alive(self) -> int:
        return sum(1 for s in self.slots if not s.done)

    def active(self) -> bool:
        return self.alive() > 0

    def _spawn(self, task: str, parent: SubAgent) -> str | None:
        """Register a new sub-agent (or None when at the concurrency cap)."""
        if self.alive() >= self.max_alive:
            return None
        from roger.agency.rollout_utils import _build_inner_fn
        self._seq += 1
        handle = f"s{self._seq}"
        session = ToolSession(prompt_backend=self.prompt_backend, policy_file=self.policy_file)
        tools, handlers = get_standard_tools(session)
        tools = tools + self.mcp_tools
        handlers = {**handlers, **self.mcp_handlers}
        slot = SubAgent(task=task, handle=handle, session=session, tools=tools,
                        tool_handlers=handlers, inner_fn=None, parent=parent, first_prompt=task)
        # Full toolset incl. recursive spawn (bound to this slot as parent) + skills.
        spawn = make_subagent_tool(self, parent=slot)
        tools.append(spawn); handlers["spawn_subagent"] = spawn
        sys_content = build_subagent_prompt(self.root)
        if self.enable_skills:
            sk = discover_skills(self.skills_root)
            if sk:
                catalog, loader = make_skill_loader(sk)
                tools.append(loader); handlers["load_skill"] = loader
                sys_content += ("\nAvailable skills (call load_skill(name) to load one's instructions):\n"
                                + catalog)
        slot.inner_fn = _build_inner_fn(
            self.tok, [t["function"]["name"] if isinstance(t, dict) else t.__name__ for t in tools])
        msgs = [{"role": "system", "content": sys_content},
                {"role": "user", "content": task + SUMMARY_SUFFIX}]
        slot.input_ids = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_tensors="pt",
            tools=tools).to(self.model.device)
        self.slots.append(slot)
        return handle

    # -- the batched turn -------------------------------------------------------------------------
    async def _one_turn(self) -> None:
        ready = [s for s in self.slots if not s.done and s.phase == "ready"]
        if not ready:
            return
        for s in ready:
            s.input_ids = torch.cat([s.input_ids, self.gen_prompt], dim=1)
        batched_generate(ready, self.model, self.processor, self.tool_tokens, self.max_new_tokens)
        await asyncio.gather(*(self._after_generate(s) for s in ready))
        # Deliver freshly-finished sub-agents to their parent (recursion) or the main agent.
        for s in [x for x in self.slots if x.done and not x.collected]:
            s.collected = True
            if s.parent is None:
                self.to_main.append((s.handle, s.result))
            else:
                self._inject_result(s.parent, s.handle, s.result)
        # Prune delivered slots so the batch stays small.
        self.slots = [s for s in self.slots if not (s.done and s.collected)]

    async def _after_generate(self, slot: SubAgent) -> None:
        """Post-generate step for one sub-agent: record, run tools, then continue / park / finish."""
        from roger.agency.rollout_utils import _old_logps, _execute_tools, _result_to_ids, _bg_msgs
        tok = self.tok; dev = self.model.device
        gen_start = slot.input_ids.shape[1]
        slot.input_ids = torch.cat([slot.input_ids, slot.new_ids.unsqueeze(0).to(dev)], dim=1)
        results = await _execute_tools(slot.new_ids, tok, slot.tool_handlers, self.tool_tokens,
                                       self.on_tool_call, self.on_tool_result)
        step_reward = max(-1.0, min(1.0, sum(reward_utils.auto_signal(r) for _, r in results)))
        slot.trajectory.append({
            "gen_start": gen_start,
            "old_logp": _old_logps(slot.step_logits, slot.masks, slot.new_ids, dev),
            "masks": slot.masks,
            "reward": step_reward,
        })
        if results:
            slot.emitted_tool_call = True
        slot.steps += 1
        pending = slot.session.pending_jobs()
        has_children = any(c.parent is slot and not c.done for c in self.slots)
        if results:
            tool_ids, _mm = _result_to_ids(results, self.processor, self.tool_res_id, dev)
            bg = _bg_msgs(slot.session, self.processor, self.tool_res_id, dev)
            parts = [slot.input_ids, tool_ids] + ([bg] if isinstance(bg, torch.Tensor) else [])
            slot.input_ids = torch.cat(parts, dim=1)
            slot.phase = "ready"
        elif pending:
            await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            bg = _bg_msgs(slot.session, self.processor, self.tool_res_id, dev)
            if isinstance(bg, torch.Tensor):
                slot.input_ids = torch.cat([slot.input_ids, bg], dim=1)
            slot.phase = "ready"
        elif has_children:
            slot.phase = "parked"          # unparked when a child result is injected
        else:
            await self._finish(slot)
        # Autonomous step budget (no interactive check-in for sub-agents).
        if not slot.done and slot.phase == "ready" and slot.steps >= self.max_steps:
            await self._finish(slot, note="(reached step limit)")

    async def _finish(self, slot: SubAgent, note: str = "") -> None:
        seg_end = len(slot.trajectory)
        answer = self.tok.decode(slot.new_ids, skip_special_tokens=True).strip()
        slot.result = answer or note or "(no answer)"
        await _grade_and_record(slot, seg_end, self.model, self.processor, self.tool_tokens, self.root)
        slot.session.terminate_jobs()
        slot.done = True

    def _inject_result(self, parent: SubAgent, handle: str, result: str) -> None:
        from roger.agency.rollout_utils import _result_to_ids
        tool_ids, _ = _result_to_ids([("spawn_subagent", f"[sub-agent {handle}] {result}")],
                                     self.processor, self.tool_res_id, self.model.device)
        parent.input_ids = torch.cat([parent.input_ids, tool_ids], dim=1)
        parent.phase = "ready"

    async def run_until_progress(self) -> list[tuple[str, str]]:
        """Advance sub-agents until a main-parented one finishes (return those), or nothing is left
        to run. Called by the main agent while parked. Empty list ⇒ no further progress possible."""
        while self.active():
            await self._one_turn()
            if self.to_main:
                out, self.to_main = self.to_main, []
                return out
            if not any((not s.done and s.phase == "ready") for s in self.slots):
                break                      # deadlock guard: only parked/done remain
        out, self.to_main = self.to_main, []
        return out


def make_subagent_tool(scheduler: "SubAgentScheduler", parent: SubAgent = None) -> Callable:
    """Return an async spawn_subagent tool bound to `scheduler`, recording `parent` as the spawner
    (None for the main agent). The docstring/signature is the model-facing schema."""
    async def spawn_subagent(task: str) -> str:
        """Spawn a sub-agent to autonomously carry out a focused, self-contained sub-task. It runs
        concurrently with any other sub-agents and reports its final findings back to you when done.
        Use this to parallelize independent parts of a larger task; returns immediately with a handle.
        Args:
            task: A complete, standalone description of the sub-task for the sub-agent to perform.
        """
        handle = scheduler._spawn(task, parent)
        if handle is None:
            return ("At capacity: the maximum number of sub-agents are already running. "
                    "Wait for a running sub-agent to finish before spawning another.")
        return f"sub-agent {handle} started; its findings will be reported back to you when it finishes."
    return spawn_subagent
