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

import os, re, shutil, time, tempfile
from ddgs import DDGS                  # keyless DuckDuckGo search + page fetch (web_search/web_fetch)
from roger.tools import shell_tools   # shell execution + policy machinery lives here
from roger.agency.path_utils import state_dir


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

# Self-eval grade [-1,1] from the most recent finish(); the rollout reads it via finish_score().
# None == no grade pending (no finish since the last task / override window closed), so this one
# var doubles as the "is there a grade the user can still override?" flag (see pending_grade()).
_finish_score: float | None = None


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
    """Copy an existing file to ~/.roger/backups/{path}.{timestamp}.bak and record for revert."""
    # ~/.roger/ is Roger's own state (memory, runs, …) — never back up or revert its contents.
    if os.path.abspath(path).startswith(state_dir() + os.sep):
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    try:
        rel = os.path.relpath(path)
    except ValueError:
        rel = os.path.basename(path)
    backup_path = os.path.join(state_dir(), "backups", f"{rel}.{ts}.bak")
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


def finish(score: float = 0.0) -> str:
    """Call when you have finished a task.
    Args:
        score: Your self-evaluation in [-1, 1] of how well the task was accomplished
               (1 = fully succeeded, 0 = unclear / partial, -1 = failed). This is the training
               reward for the task, so grade truthfully.
    """
    # Coerce defensively: the decoding constraint doesn't type-check args.
    global _finish_score
    try:
        _finish_score = max(-1.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        _finish_score = 0.0
    return "(done)"


def finish_score() -> float | None:
    """The most recent finish()'s self-eval grade, or None when no finish is overridable yet.
    None is the 'nothing pending' flag driving the /grade ghost text; reward math reads it as
    `finish_score() or 0.0`."""
    return _finish_score


def set_user_grade(score) -> float | None:
    """User override of the pending grade. No-op -> None when nothing's pending or the value is
    unparseable (so an accidental ghost-text accept can't grade). Leaves it pending on success so
    the hint keeps offering re-override until the next task."""
    global _finish_score
    if _finish_score is None:
        return None
    try:
        _finish_score = max(-1.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return None
    return _finish_score


def clear_grade() -> None:
    """Close the override window at a fresh task segment so a stale grade hint doesn't linger."""
    global _finish_score
    _finish_score = None


# ---------------------------------------------------------------------------
# Web (keyless DuckDuckGo — ddgs handles request headers/UA rotation internally)
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 10) -> str:
    """Search the web (DuckDuckGo) and return ranked results as title / url / snippet.
    Args:
        query: The search query.
        max_results: Maximum number of results to return.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:   # network/backend errors → string the agent can react to
        return f"Search error: {e}"
    if not results:
        return f"No results for: {query}"
    return "\n\n".join(
        f"{r.get('title', '').strip()}\n{r.get('href', '').strip()}\n{r.get('body', '').strip()}"
        for r in results
    )


def web_fetch(url: str) -> str:
    """Fetch a web page and return its main content as readable markdown.
    Args:
        url: The URL to fetch.
    """
    try:
        result = DDGS().extract(url)   # default fmt=text_markdown → {"url", "content"}
    except Exception as e:
        return f"Fetch error: {e}"
    content = result.get("content") if isinstance(result, dict) else result
    return content or f"No content extracted from {url}"


# ---------------------------------------------------------------------------
# End-of-rollout revert
# ---------------------------------------------------------------------------

def pending_backups() -> list[tuple[str, str]]:
    """Current (original_path, backup_path) pairs awaiting a revert decision; empty when nothing changed."""
    return list(_backups)


def apply_revert(answer: str) -> int:
    """Revert backed-up files per `answer` ('all' / '1,3' / 'none'); return the count reverted.
    Only the reverted entries are popped from _backups — a partial revert keeps the rest available
    for a later offer (so the user can revert one file now and the others next time). 'none' discards
    everything; an answer with no recognised index is a no-op (doesn't clear), so an accidental
    ghost-text accept can't wipe the list."""
    answer = (answer or "").strip().lower()
    if answer == "none":
        _backups.clear(); return 0
    if answer in ("", "all"):
        indices = set(range(len(_backups)))
    else:
        # Keep only integer tokens (split on commas/whitespace); ignore anything else.
        indices = {int(t) - 1 for t in re.split(r"[,\s]+", answer) if t.isdigit()}
        if not indices:
            return 0
    n_reverted = 0
    kept: list[tuple[str, str]] = []
    for idx, (orig, bak) in enumerate(_backups):
        if idx in indices:
            try: shutil.copy2(bak, orig); n_reverted += 1
            except OSError: kept.append((orig, bak))   # restore failed → keep so the user can retry
        else:
            kept.append((orig, bak))                   # not selected → still pending
    _backups[:] = kept
    return n_reverted


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
    global _prompt_backend, _policy_file, _finish_score
    if prompt_backend is not None:
        _prompt_backend = prompt_backend
    _policy_file = policy_file

    # Reset per-rollout state
    _backups.clear()
    _finish_score = None

    # Configure shell_tools (sets its own prompt/policy globals, resets _jobs)
    shell_tools.configure(prompt_backend=prompt_backend, policy_file=policy_file)

    shell = [shell_tools.run_command, shell_tools.stop_command, shell_tools.check_command]
    file  = [write_file, edit_file]
    web   = [web_search, web_fetch]
    misc  = [prompt_user, finish]
    tools = shell + file + web + misc
    handlers = {fn.__name__: fn for fn in tools}
    return tools, handlers
