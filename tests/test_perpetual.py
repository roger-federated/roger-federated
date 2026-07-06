"""Tests for perpetual (/perpetual) agent loops + the reason-then-force seed helper.
Run with:  conda run -n roger python -m pytest tests/test_perpetual.py

No model is loaded: every unit here exercises the pure helpers in rollout_utils (command parsing,
the varied continuation seed, the soft-prefill seed builder, the force-tool-after-think-close
constraint) plus the prompt_user suppression seam in get_standard_tools."""
import types, torch
import roger.agency.rollout_utils as r
from roger.tools.std_tools import get_standard_tools


# --- /perpetual command parsing -------------------------------------------------------------

def test_parse_perpetual_strips_marker():
    assert r._parse_perpetual("/perpetual watch the site") == (True, "watch the site")
    assert r._parse_perpetual("/loop keep improving") == (True, "keep improving")   # alias
    assert r._parse_perpetual("  /PerPetual  do X  ") == (True, "do X")             # case + whitespace
    assert r._parse_perpetual("normal task") == (False, "normal task")             # untouched
    # A bare marker yields an empty task (caller still runs one normal turn first).
    assert r._parse_perpetual("/perpetual") == (True, "")


# --- varied, comprehensive continuation seed ------------------------------------------------

# autonomy / completion / error-recovery / adjacent / untried / sleep-and-recheck — every beat, every variant
_ASPECTS = ("autonom", "satisf", "error", "adjacent", "tried", "sleep")

def test_seed_is_randomly_varied():
    draws = {r._perpetual_seed_text("t", "web_fetch", 0) for _ in range(50)}
    assert len(draws) >= 3   # random.choice over the pool → not the same opener every time

def test_every_seed_variant_covers_all_beats():
    # Sample enough to surface all variants; each must touch every beat (no dropped coverage).
    for _ in range(200):
        s = r._perpetual_seed_text("my task", "run_command", 1).lower()
        for a in _ASPECTS:
            assert a in s, f"beat {a!r} missing from: {s}"

def test_seed_restates_task_and_grounds_on_last_tool_and_paces_on_idle():
    assert "my task" in r._perpetual_seed_text("my task", None, 0)          # re-anchors the goal every time
    assert "`web_fetch`" in r._perpetual_seed_text("t", "web_fetch", 0)
    assert "last used" not in r._perpetual_seed_text("t", None, 0)          # no tool yet → no claim
    assert "sleep 10" in r._perpetual_seed_text("t", None, 1)               # idle escalates pacing
    assert "sleep 60" in r._perpetual_seed_text("t", None, 9)               # capped at 60s

def test_seed_subagent_nudge_gated_on_availability():
    # The spawn nudge only appears when spawning is actually available (scheduler present).
    for _ in range(50):
        assert "sub-agent" in r._perpetual_seed_text("t", None, 0, can_spawn=True)
        assert "sub-agent" not in r._perpetual_seed_text("t", None, 0, can_spawn=False)


# --- soft-prefill seed builder --------------------------------------------------------------

class _FakeTok:
    """Minimal tokenizer: encode(text) → list of char codes; carries an eos id."""
    eos_token_id = 999
    def encode(self, text, add_special_tokens=False): return [ord(c) for c in text]

def test_build_seed_reasoning_model_opens_think_channel():
    tok, think, tool = _FakeTok(), (10, 11), (100, 101)
    seed = r._build_seed(tok, "hi", think, tool, force_tool=True)
    assert seed[0] == 10                       # think-open first
    assert 100 not in seed                     # tool-open NOT spliced (fires after the model closes think)

def test_build_seed_non_reasoning_fallback():
    tok, tool = _FakeTok(), (100, 101)
    # No think channel → mandatory seed splices the tool-open immediately (old behaviour preserved).
    assert r._build_seed(tok, "hi", None, tool, force_tool=True)[-1] == 100
    # Unforced (perpetual) seed on a plain model is just the body, no tool-open.
    assert 100 not in r._build_seed(tok, "hi", None, tool, force_tool=False)


# --- reason-then-force constraint ------------------------------------------------------------

def test_force_tool_only_after_think_close():
    tok, tool, think_close = _FakeTok(), (100, 101), 50
    inner = lambda *_: [7, 8]                   # would run only inside a tool block; must NOT here

    def allowed(ids):
        return r._allowed_tokens(inner, tok, tool, torch.tensor([ids]), 0,
                                 think_close=think_close, force_tool=True)

    assert allowed([10, 20, 30]) is None                 # still reasoning, no think-close → free
    assert allowed([10, 20, 50]) == [100]                # think-close seen → force the tool-open once
    # Once the call has started (tool-open present), we take the JSON path, never re-force the open.
    assert allowed([50, 100, 60]) == [7, 8]

def test_no_force_when_flag_off():
    tok, tool, think_close = _FakeTok(), (100, 101), 50
    inner = lambda *_: [7, 8]
    out = r._allowed_tokens(inner, tok, tool, torch.tensor([[10, 50]]), 0,
                            think_close=think_close, force_tool=False)
    assert out is None                                   # think-close present but forcing disabled → free


# --- prompt_user suppression seam ------------------------------------------------------------

def test_prompt_user_toggle_in_standard_tools():
    _, on = get_standard_tools(types_session(), expose_prompt_user=True)
    assert "prompt_user" in on
    _, off = get_standard_tools(types_session(), expose_prompt_user=False)
    assert "prompt_user" not in off
    # Suppressing prompt_user must not drop the ordinary tools.
    assert {"write_file", "edit_file", "run_command"} <= set(off)


def types_session():
    from roger.tools.session import ToolSession
    return ToolSession(prompt_backend=lambda *_a, **_k: "")
