"""Tests for reward utilities and std_tools reward-related behaviour.
Run with:  PYTHONPATH=src python -m pytest tests/test_rewards.py
"""

import sys
import roger.training.reward_utils as reward_utils
from roger.training.reward_utils import auto_signal

# ---------------------------------------------------------------------------
# auto_signal — pure function
# ---------------------------------------------------------------------------

def test_auto_signal_clean():
    assert auto_signal("Wrote 42 bytes to foo.txt") == 0.0
    assert auto_signal("exit 0\nsome output") == 0.0
    assert auto_signal(None) == 0.0
    print("PASS test_auto_signal_clean")

def test_auto_signal_exit_code():
    assert auto_signal("exit 1\nfailed") < 0
    assert auto_signal("exit 127\ncommand not found") < 0
    print("PASS test_auto_signal_exit_code")

def test_auto_signal_error_str():
    assert auto_signal("Error: file not found") < 0
    assert auto_signal("Blocked by policy: rm -rf /") < 0
    assert auto_signal("File not found: /some/path") < 0       # actual std_tools return
    assert auto_signal("Permission denied: /etc/shadow") < 0
    assert auto_signal("Command timed out after 30s") < 0      # actual std_tools return
    print("PASS test_auto_signal_error_str")

def test_auto_signal_cmd_reject():
    # run_command returns this string on rejection; should carry W_CMD_REJ weight
    r = auto_signal("Command rejected by user: rm -rf /tmp/x")
    assert r < 0
    assert abs(r) >= reward_utils.W_CMD_REJ - 1e-9
    print("PASS test_auto_signal_cmd_reject")

def test_auto_signal_clip():
    # Even if multiple patterns fire simultaneously the result stays in [-1, 1]
    r = auto_signal("Command rejected by user: exit 1\nError: also bad")
    assert -1.0 <= r <= 1.0
    print("PASS test_auto_signal_clip")

# ---------------------------------------------------------------------------
# std_tools: pending_backups / apply_revert
# ---------------------------------------------------------------------------

def test_apply_revert_no_backups():
    import roger.tools.std_tools as std_tools
    std_tools._backups.clear()
    assert std_tools.pending_backups() == []
    assert std_tools.apply_revert("all") == 0
    print("PASS test_apply_revert_no_backups")

def test_apply_revert_none_clears():
    import roger.tools.std_tools as std_tools
    std_tools._backups = [("/fake/orig.txt", "/fake/orig.txt.bak")]
    assert len(std_tools.pending_backups()) == 1
    assert std_tools.apply_revert("none") == 0
    assert std_tools.pending_backups() == []      # decision is final → backups cleared
    print("PASS test_apply_revert_none_clears")

def test_apply_revert_all_restores():
    import os, tempfile
    import roger.tools.std_tools as std_tools
    d = tempfile.mkdtemp()
    orig, bak = os.path.join(d, "f.txt"), os.path.join(d, "f.bak")
    open(orig, "w").write("NEW"); open(bak, "w").write("OLD")
    std_tools._backups = [(orig, bak)]
    assert std_tools.apply_revert("all") == 1
    assert open(orig).read() == "OLD" and std_tools.pending_backups() == []
    print("PASS test_apply_revert_all_restores")

# ---------------------------------------------------------------------------
# std_tools: maxsteps_checkin
# ---------------------------------------------------------------------------

def test_maxsteps_checkin_options():
    import roger.tools.std_tools as std_tools
    for inp, expected in [("1", "continue"), ("2", "abort"), ("", "continue")]:
        std_tools._prompt_backend = lambda q, _i=inp: _i
        action, fb = std_tools.maxsteps_checkin()
        assert action == expected, f"input {inp!r} → {action}"
    responses = iter(["3", "try a different approach"])
    std_tools._prompt_backend = lambda q: next(responses)
    action, fb = std_tools.maxsteps_checkin()
    assert action == "feedback" and fb == "try a different approach"
    print("PASS test_maxsteps_checkin_options")

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_auto_signal_clean,
        test_auto_signal_exit_code,
        test_auto_signal_error_str,
        test_auto_signal_cmd_reject,
        test_auto_signal_clip,
        test_apply_revert_no_backups,
        test_apply_revert_none_clears,
        test_apply_revert_all_restores,
        test_maxsteps_checkin_options,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(failed)
