"""Standard tools for the agent rollout loop.

Each tool is a plain callable with Google-style docstrings so that HF's
apply_chat_template(tools=...) can auto-generate JSON schemas from the
signature + docstring.  All tools return str and never raise — errors
become descriptive strings the agent can reason about and retry.

Only content-emitting operations (write_file, edit_file) are kept as tools;
consuming/moving/computing operations are covered by run_command idioms
described in the system prompt.

Usage:
    tools, handlers = get_standard_tools()
    await rollout(..., tools=tools, tool_handlers=handlers)
"""

import os, shutil, time, tempfile
import shell_tools   # shell execution + policy machinery lives here


# ---------------------------------------------------------------------------
# Module-level per-rollout state
# ---------------------------------------------------------------------------

def _default_prompt(question: str) -> str:
    try:
        return input(question + " ")
    except EOFError:
        return "(no response — user input unavailable)"
    except KeyboardInterrupt:
        return "(user cancelled)"

_prompt_backend = _default_prompt
_policy_file: str = "command_policy.txt"   # kept here so write_file/_is_protected can read it

# Write backups — (original_path, backup_path) for end-of-rollout revert
_backups: list[tuple[str, str]] = []


# ---------------------------------------------------------------------------
# File tools (emitters only — reading/searching goes through run_command)
# ---------------------------------------------------------------------------

def write_file(path: str, content: str, append: bool = False) -> str:
    """Write or append content to a file, creating parent directories as needed.
    Args:
        path: Absolute or relative path to the target file.
        content: The text content to write.
        append: If true, append to existing file instead of overwriting.
    """
    if shell_tools._is_protected(path):
        return f"Refused: {path} is the guardrail policy file and cannot be modified by the agent."
    try:
        if not append and os.path.isfile(path):
            _backup_file(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            n = f.write(content)
        verb = "Appended" if append else "Wrote"
        return f"{verb} {n} bytes to {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except OSError as e:
        return f"Error writing file: {e}"


def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    """Replace an exact substring in a file. View the file first (e.g. `cat`/`Get-Content`) to get exact text.
    Args:
        path: File to edit.
        old: Exact text to find (must be unique in the file unless replace_all is set).
        new: Replacement text.
        replace_all: If true, replace every occurrence instead of requiring uniqueness.
    """
    if shell_tools._is_protected(path):
        return f"Refused: {path} is the guardrail policy file and cannot be modified by the agent."
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except OSError as e:
        return f"Error reading file: {e}"

    n = content.count(old)
    if n == 0:
        return f"No occurrence of the target text in {path}"
    if not replace_all and n > 1:
        return f"Target text is not unique ({n} matches); add more context or set replace_all"

    new_content = content.replace(old, new) if replace_all else content.replace(old, new, 1)
    _backup_file(path)
    # Atomic write: temp file in same dir, then os.replace
    try:
        parent = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)
            os.replace(tmp, path)
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise
    except PermissionError:
        return f"Permission denied: {path}"
    except OSError as e:
        return f"Error writing file: {e}"

    replaced = n if replace_all else 1
    return f"Replaced {replaced} occurrence(s) in {path}"


def _backup_file(path: str) -> None:
    """Copy an existing file to .roger/backups/{path}.{timestamp}.bak and record for revert."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    try:
        rel = os.path.relpath(path)
    except ValueError:
        rel = os.path.basename(path)
    backup_path = os.path.join(".roger", "backups", f"{rel}.{ts}.bak")
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy2(path, backup_path)
    _backups.append((os.path.abspath(path), os.path.abspath(backup_path)))


# ---------------------------------------------------------------------------
# User interaction
# ---------------------------------------------------------------------------

def prompt_user(question: str) -> str:
    """Ask the user a question and return their response.
    Args:
        question: The question or message to display to the user.
    """
    return _prompt_backend(question)


def finish(message: str = "") -> str:
    """Signal that the task is complete. Optionally provide a final answer or summary.
    Args:
        message: Final answer / summary to show the user. May be empty.
    """
    # The rollout loop detects this call by name and terminates after processing it.
    return message or "(done)"


# ---------------------------------------------------------------------------
# End-of-rollout revert
# ---------------------------------------------------------------------------

def offer_revert(prompt_fn=None) -> tuple[bool, int]:
    """Prompt the user to revert files overwritten during the rollout.
    Returns (was_prompted, n_reverted): was_prompted is False when no backups existed."""
    if not _backups:
        return False, 0
    fn = prompt_fn or _prompt_backend
    listing = "\n".join(f"  {i+1}. {orig}" for i, (orig, _bak) in enumerate(_backups))
    answer = fn(f"The following files were overwritten and backed up:\n{listing}\nRevert? [all / 1,2,... / none]")
    answer = answer.strip().lower()

    if answer == "none" or not answer:
        _backups.clear(); return True, 0
    if answer == "all":
        indices = list(range(len(_backups)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in answer.split(",")]
        except ValueError:
            _backups.clear(); return True, 0

    n_reverted = 0
    for idx in indices:
        if 0 <= idx < len(_backups):
            orig, bak = _backups[idx]
            try: shutil.copy2(bak, orig); n_reverted += 1
            except OSError: pass
    _backups.clear()
    return True, n_reverted


# ---------------------------------------------------------------------------
# Max-steps check-in (called directly by rollout; not an agent tool)
# ---------------------------------------------------------------------------

def maxsteps_checkin(prompt_fn=None) -> tuple[str, str]:
    """Present the three-option max-steps check-in to the user.
    Returns (action, feedback_text): action in {'continue', 'abort', 'feedback'}."""
    fn = prompt_fn or _prompt_backend
    answer = fn(
        "Max steps reached.\n"
        "  [1] Continue iterating\n"
        "  [2] Abort\n"
        "  [3] Continue with feedback\n"
        "Choice [1/2/3]:"
    ).strip()
    if answer in ("2", "abort"):
        return "abort", ""
    if answer in ("3", "feedback", "3 continue with feedback"):
        text = fn("Feedback:").strip()
        return "feedback", text
    return "continue", ""


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def get_standard_tools(prompt_backend=None, policy_file="command_policy.txt") -> tuple[list, dict]:
    """Return (tools_list, tool_handlers) for use with rollout().
    Args:
        prompt_backend: Optional callable(question: str) -> str replacing input().
        policy_file: Path to the command policy file for run_command.
    """
    global _prompt_backend, _policy_file
    if prompt_backend is not None:
        _prompt_backend = prompt_backend
    _policy_file = policy_file

    # Reset per-rollout state
    _backups.clear()

    # Configure shell_tools (sets its own prompt/policy globals, resets _jobs, and
    # ensures .roger/ is git-ignored)
    shell_tools.configure(prompt_backend=prompt_backend, policy_file=policy_file)

    shell = [shell_tools.run_command, shell_tools.stop_command, shell_tools.check_command]
    file  = [write_file, edit_file]
    misc  = [prompt_user, finish]
    tools = shell + file + misc
    handlers = {fn.__name__: fn for fn in tools}
    return tools, handlers
