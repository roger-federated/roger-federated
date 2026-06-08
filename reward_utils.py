"""Reward utilities for REINFORCE++ rollouts.

auto_signal() is a pure function called per step by the rollout loop.
Terminal signals (revert, outcome) are computed by the rollout and broadcast
over all trajectory steps at episode end.
"""

import re

# ---------------------------------------------------------------------------
# Weights — sign and relative magnitude matter; REINFORCE++ z-normalizes
# advantages across the batch so the absolute scale is forgiving.
# ---------------------------------------------------------------------------
W_TERMINAL  = 1.0   # good/bad outcome at episode end
W_ABORT     = 1.0   # user aborts via max-steps check-in
W_CONTINUE  = 0.1   # user confirms "on track" at max-steps check-in
W_FEEDBACK  = 0.2   # user provides corrective feedback
W_REVERT    = 0.5   # per file the user reverts
W_CMD_REJ   = 0.3   # user rejects a confirm-policy command
W_EXIT      = 0.1   # nonzero shell exit code
W_ERROR     = 0.1   # tool result is an error string

_SIGNALS = [
    (r"^Command rejected by user",                                          -W_CMD_REJ),
    (r"^exit [1-9]\d*\b",                                                   -W_EXIT),
    (r"Error:|Blocked|not found|Permission denied|timed out",               -W_ERROR),
]
_COMPILED = [(re.compile(pat, re.IGNORECASE), w) for pat, w in _SIGNALS]


def auto_signal(result) -> float:
    """Return a reward signal for a single tool result string.
    Catches command rejections, nonzero exit codes, and error-string prefixes.
    Returns 0.0 for non-string results or clean outputs.
    """
    if not isinstance(result, str):
        return 0.0
    total = sum(w for pat, w in _COMPILED if pat.search(result))
    return max(-1.0, min(1.0, total))
