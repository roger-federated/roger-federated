"""signals.py — verifiable per-step rewards read off the tool results a client sends back.

The legacy harness ran tools itself and scored each result with training/reward_utils.auto_signal
(nonzero exit code, error strings, user rejection), summed per generation turn. Behind the wrapper the
*client* runs the tools, and their results reach us as `role: "tool"` messages (chat) or
`function_call_output` items (responses) inside the next request — so the same signal is recovered by
reading the history. Ported here rather than imported: the legacy package is slated for removal and the
runtime stays torch- and harness-free.
"""
import json, re

# Same weights as the legacy reward_utils: sign and relative size matter, REINFORCE++ z-normalises the
# advantages so the absolute scale is forgiving.
W_CMD_REJ = 0.3     # user rejected a command
W_EXIT = 0.1        # nonzero exit code
W_ERROR = 0.1       # result reads as an error

# The result text is the client harness's format, not roger's, so the exit-code pattern takes the common
# phrasings: roger's own `exit 1`, `Exit code: 1`, `exit status 2`, `exited with code 1`, `[exit code 1]`.
# Only a nonzero code fires, so `exit code 0` headers stay clean.
_SIGNALS = [
    (r"^Command rejected by user", -W_CMD_REJ),
    (r"\bexit(?:ed)?(?: with)?(?: code| status)?\s*[:=]?\s*[1-9]\d*\b", -W_EXIT),
    (r"Error:|Blocked|not found|Permission denied|timed out", -W_ERROR),
]
_COMPILED = [(re.compile(pat, re.IGNORECASE | re.MULTILINE), w) for pat, w in _SIGNALS]


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _text(content) -> str:
    # Tool content may be a part list (OpenAI allows [{type: text, text: …}]); only the text is scoreable.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else json.dumps(content)


def score(text: str) -> float:
    return _clamp(sum(w for pat, w in _COMPILED if pat.search(text)))


def _is_step(m: dict) -> bool:
    return m.get("role") == "assistant" or m.get("type") == "function_call"


def step_rewards(items) -> dict[str, float]:
    """{index of the assistant turn: clamped sum of its tool results' signals}, nonzero steps only.
    A step is the wire equivalent of a legacy generation turn: a run of assistant-side items (one chat
    assistant message, or a responses-API run of `function_call` items) anchored at its first index;
    the tool results that follow belong to it."""
    if not isinstance(items, list):
        return {}                                          # responses `input` may be a bare string
    sums: dict[int, float] = {}
    anchor, prev_step = None, False
    for i, m in enumerate(items):
        if not isinstance(m, dict):
            prev_step = False
            continue
        if _is_step(m):
            if not prev_step:
                anchor = i
            prev_step = True
            continue
        prev_step = False
        if anchor is None:
            continue                                       # a result with no call in view: nothing to credit
        if m.get("role") == "tool":
            sums[anchor] = sums.get(anchor, 0.0) + score(_text(m.get("content")))
        elif m.get("type") == "function_call_output":
            sums[anchor] = sums.get(anchor, 0.0) + score(_text(m.get("output")))
    # String keys: this lands in JSON, where keys are strings anyway — keep in-memory and on-disk equal.
    return {str(i): _clamp(v) for i, v in sums.items() if v != 0.0}
