"""Token-stream integrity tests: the rollout's incrementally-built stream must be byte-identical
to the canonical chat-template render — no doubled tool-response openers, no spurious turn cues
after tool results, no missing turn-close glue. Run with:
    conda run -n roger python -m pytest tests/test_stream.py

Helper tests use a fake tokenizer; the end-to-end reconstruction tests use the real gemma-4
tokenizer from the local HF cache (skipped when absent — they never download)."""
import torch
import pytest
from roger.agency.rollout_utils import _close_turn
from roger.agency.subagents import _cut_at_stop

TURN_END = [106, 107]          # <turn|> \n (gemma-4 shaped)


# --- _close_turn: splice the never-generated close/glue --------------------------------------

def _ids(*toks):
    return torch.tensor([list(toks)], dtype=torch.long)

def test_close_turn_adds_glue_after_stop_at_close():
    # generation halts AT <turn|>; the template's trailing \n is never emitted
    assert _close_turn(_ids(5, 106), TURN_END)[0].tolist() == [5, 106, 107]

def test_close_turn_full_close_on_bare_turn():
    # eos-trimmed or interrupted turn: no close at all → append the whole tail
    assert _close_turn(_ids(5, 9), TURN_END)[0].tolist() == [5, 9, 106, 107]

def test_close_turn_noop_when_already_closed():
    ids = _ids(5, 106, 107)
    assert _close_turn(ids, TURN_END) is ids     # idempotent: safe to call at every boundary

def test_close_turn_noop_without_probe():
    ids = _ids(5, 9)
    assert _close_turn(ids, []) is ids           # probe failed → old behaviour


# --- _cut_at_stop: sub-agent row trimming ------------------------------------------------------

STOP, DROP = {1, 50, 106}, {1, 50}

def test_cut_keeps_turn_close_drops_eos_and_tool_res():
    assert _cut_at_stop([7, 8, 106, 0, 0], STOP, DROP) == ([7, 8, 106], 106)  # close kept, pads gone
    assert _cut_at_stop([7, 8, 50], STOP, DROP) == ([7, 8], 50)   # tool delta re-supplies the opener
    assert _cut_at_stop([7, 8, 1], STOP, DROP) == ([7, 8], 1)     # bare eos dropped
    assert _cut_at_stop([7, 8], STOP, DROP) == ([7, 8], None)     # ran to length: nothing to cut


# --- end-to-end stream reconstruction against the canonical template render -------------------

MID = "google/gemma-4-E2B-it"

@pytest.fixture(scope="module")
def proc():
    from transformers import AutoProcessor
    try:
        return AutoProcessor.from_pretrained(MID, local_files_only=True)
    except Exception:
        pytest.skip(f"{MID} not in the local HF cache")

@pytest.fixture(scope="module")
def probes(proc):
    from roger.loading.model_setup import (find_gen_prompt, find_gen_prompt_after_tool,
                                           find_turn_end, find_tool_res_id)
    tok = proc.tokenizer
    return {"tok": tok, "gp": find_gen_prompt(tok), "gpt": find_gen_prompt_after_tool(tok),
            "turn_end": find_turn_end(tok), "tr": find_tool_res_id(tok), "eos": tok.eos_token_id}

SYS  = {"role": "system", "content": "You are a test."}
U1   = {"role": "user", "content": "hi"}
CALL = {"role": "assistant", "tool_calls": [
        {"id": "0", "type": "function", "function": {"name": "run_command", "arguments": {"command": "ls"}}}]}
RES  = {"role": "tool", "tool_call_id": "0", "content": [{"type": "text", "text": "file.txt"}]}
ANS  = {"role": "assistant", "content": "Done."}
U2   = {"role": "user", "content": "thanks"}
ASST_MSG = [{"role": "assistant", "tool_calls": [
             {"id": "0", "type": "function", "function": {"name": "_", "arguments": {}}}]}]


def _T(tok, msgs, gen=False):
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)["input_ids"]

def _emitted(p, prefix_msgs, msgs, cue):
    """What generation actually emits for this turn: the template continuation truncated at the
    first stop token (HF generate halts there, inclusive)."""
    base = _T(p["tok"], prefix_msgs) + cue
    full = _T(p["tok"], msgs)
    assert full[:len(base)] == base, "template render is not prefix-stable"
    cont = full[len(base):]
    stops = {p["eos"], p["tr"]} | ({p["turn_end"][0]} if p["turn_end"] else set())
    for i, t in enumerate(cont):
        if t in stops:
            return cont[:i + 1]
    return cont

def _sim_turn(p, input_ids, gen_toks):
    """The loop's post-generate step: trailing eos / tool-res-opener trim."""
    seq = torch.cat([input_ids, torch.tensor([gen_toks])], dim=1)
    last = seq[0, -1].item()
    return (seq[:, :-1] if last in (p["eos"], p["tr"]) else seq), last == p["tr"]


def test_tool_call_flow_matches_template(proc, probes):
    """tool-call turn → results (no doubled opener, after-tool cue) → answer → user turn."""
    from roger.agency.rollout_utils import _result_to_ids, _user_turn_delta
    p = probes
    ids = torch.tensor([_T(p["tok"], [SYS, U1])])
    ids = torch.cat([ids, torch.tensor([p["gp"]])], dim=1)
    gen_ids, awaiting = _sim_turn(p, ids, _emitted(p, [SYS, U1], [SYS, U1, CALL], p["gp"]))
    assert awaiting                                            # stopped at the tool-response opener
    tool_ids, _ = _result_to_ids([("run_command", "file.txt")], proc, p["tr"], "cpu")
    ids = torch.cat([gen_ids, tool_ids, torch.tensor([p["gpt"]])], dim=1)
    gen_ids, awaiting = _sim_turn(p, ids, _emitted(p, [SYS, U1, CALL, RES], [SYS, U1, CALL, RES, ANS], p["gpt"]))
    assert not awaiting
    gen_ids = _close_turn(gen_ids, p["turn_end"])              # task-done boundary
    ids = torch.cat([gen_ids, _user_turn_delta(p["tok"], ASST_MSG, "thanks", "cpu"),
                     torch.tensor([p["gp"]])], dim=1)
    assert ids[0].tolist() == _T(p["tok"], [SYS, U1, CALL, RES, ANS, U2], gen=True)


def test_eos_stop_flow_matches_template(proc, probes):
    """A turn ending in raw <eos> (trimmed) still yields a canonically-closed boundary."""
    from roger.agency.rollout_utils import _user_turn_delta
    p = probes
    ids = torch.tensor([_T(p["tok"], [SYS, U1])])
    ids = torch.cat([ids, torch.tensor([p["gp"]])], dim=1)
    g = _emitted(p, [SYS, U1], [SYS, U1, ANS], p["gp"])[:-1] + [p["eos"]]
    gen_ids, _ = _sim_turn(p, ids, g)
    gen_ids = _close_turn(gen_ids, p["turn_end"])
    ids = torch.cat([gen_ids, _user_turn_delta(p["tok"], ASST_MSG, "thanks", "cpu"),
                     torch.tensor([p["gp"]])], dim=1)
    assert ids[0].tolist() == _T(p["tok"], [SYS, U1, ANS, U2], gen=True)


def test_multi_call_flow_matches_template(proc, probes):
    """Two back-to-back calls: one trimmed opener, per-response blocks, after-tool cue."""
    from roger.agency.rollout_utils import _result_to_ids
    p = probes
    ac2 = {"role": "assistant", "tool_calls": [
        {"id": "0", "type": "function", "function": {"name": "f", "arguments": {}}},
        {"id": "1", "type": "function", "function": {"name": "g", "arguments": {}}}]}
    tr0 = {"role": "tool", "tool_call_id": "0", "content": [{"type": "text", "text": "r0"}]}
    tr1 = {"role": "tool", "tool_call_id": "1", "content": [{"type": "text", "text": "r1"}]}
    ids = torch.tensor([_T(p["tok"], [SYS, U1])])
    ids = torch.cat([ids, torch.tensor([p["gp"]])], dim=1)
    gen_ids, awaiting = _sim_turn(p, ids, _emitted(p, [SYS, U1], [SYS, U1, ac2], p["gp"]))
    assert awaiting
    tool_ids, _ = _result_to_ids([("f", "r0"), ("g", "r1")], proc, p["tr"], "cpu")
    ids = torch.cat([gen_ids, tool_ids, torch.tensor([p["gpt"]])], dim=1)
    assert ids[0].tolist() == _T(p["tok"], [SYS, U1, ac2, tr0, tr1], gen=True)


def test_think_prefix_matches_canonical_header(proc, probes):
    """find_think_prefix must recover exactly what the template puts between the think-open
    delimiter and the reasoning text (gemma-4: 'thought\\n'), so seeded thoughts are canonical."""
    from roger.loading.model_setup import find_think_prefix, find_think_tokens
    p = probes
    th = find_think_tokens(p["tok"])
    prefix = find_think_prefix(p["tok"])
    assert th is not None and prefix, "gemma-4 has a reasoning channel with a header"
    marker = "QQXYZQ unique reasoning"
    turn = _T(p["tok"], [U1, {"role": "assistant", "reasoning_content": marker,
                              "tool_calls": [{"id": "0", "type": "function",
                                              "function": {"name": "f", "arguments": {}}}]}])
    body = p["tok"].encode(marker, add_special_tokens=False)
    i = turn.index(th[0])
    assert turn[i + 1:i + 1 + len(prefix) + len(body)] == prefix + body


def test_load_tools_suggests_server_tools():
    """Passing an MCP server prefix (not a tool name) must return the tools under it, not a
    dead-end error (regression: the model called load_tools(['mcp__canva']) and got stuck)."""
    import json as _json
    from roger.agency.rollout_utils import _make_tool_loader
    def _t(name):
        d = {"type": "function", "function": {"name": name, "description": "d", "parameters": {}}}
        return d
    _, load_tools = _make_tool_loader([_t("mcp__canva__create_design"), _t("mcp__canva__get_asset"),
                                       _t("mcp__twitter__post")])
    out = _json.loads(load_tools(["mcp__canva"]))
    assert out[0]["error"] and set(out[0]["did_you_mean"]) == {"mcp__canva__create_design",
                                                               "mcp__canva__get_asset"}
    out = _json.loads(load_tools(["mcp__twitter__post"]))   # exact name still loads the schema
    assert out[0]["function"]["name"] == "mcp__twitter__post" and "error" not in out[0]


def test_subagent_first_turn_has_single_cue(proc, probes):
    """_spawn templates without a gen cue; _one_turn adds exactly one (regression: it was doubled)."""
    p = probes
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    base = _T(p["tok"], msgs)                       # what _spawn now builds (no gen prompt)
    once = base + p["gp"]                           # + _one_turn's single cue
    assert once == _T(p["tok"], msgs, gen=True)     # exactly the canonical gen-ready render
