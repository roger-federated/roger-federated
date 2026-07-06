"""Standard tools for the agent rollout loop.

Each tool is a plain callable with Google-style docstrings so that HF's
apply_chat_template(tools=...) can auto-generate JSON schemas from the
signature + docstring.  All tools return str and never raise — errors
become descriptive strings the agent can reason about and retry.

Only content-emitting operations (write_file, edit_file) are kept as tools;
consuming/moving/computing operations are covered by run_command idioms
described in the system prompt.

Stateful tools (file writes, shell jobs, prompts) are bound to a ToolSession:
    tools, handlers = get_standard_tools(session)
    await rollout(..., tools=tools, tool_handlers=handlers)
so concurrent sub-agents never share backups / a job registry / a grade window.
Grade + backup/revert state now live on ToolSession (see tools/session.py).
"""

import os, shutil, time, tempfile
from ddgs import DDGS                  # keyless DuckDuckGo search + page fetch (web_search/web_fetch)
from roger.tools import shell_tools   # shell execution + policy machinery lives here
from roger.agency.path_utils import state_dir


# ---------------------------------------------------------------------------
# File tools (emitters only — reading/searching goes through run_command).
# The impls take the ToolSession explicitly; get_standard_tools wraps them in
# schema-carrying closures bound to one session.
# ---------------------------------------------------------------------------

def _write_file(session, path: str, content: str, append: bool) -> str:
    if shell_tools._is_protected(path, session.policy_file):
        return f"Refused: {path} is the guardrail policy file and cannot be modified by the agent."
    try:
        if not append and os.path.isfile(path):
            _backup_file(session, path)
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


def _edit_file(session, path: str, old: str, new: str, replace_all: bool) -> str:
    if shell_tools._is_protected(path, session.policy_file):
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
    _backup_file(session, path)
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


def _backup_file(session, path: str) -> None:
    """Copy an existing file to ~/.roger/backups/{path}.{timestamp}.bak and record it on the session."""
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
    session.backups.append((os.path.abspath(path), os.path.abspath(backup_path)))


# ---------------------------------------------------------------------------
# Web (keyless DuckDuckGo — stateless; the same callables are shared by every agent)
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
# Max-steps check-in (called directly by rollout; not an agent tool)
# ---------------------------------------------------------------------------

def maxsteps_checkin(prompt_fn) -> tuple[str, str]:
    """Present the three-option max-steps check-in to the user via prompt_fn.
    Returns (action, feedback_text): action in {'continue', 'abort', 'feedback'}."""
    answer = prompt_fn(
        "Max steps reached.\n"
        "  [1] Continue iterating\n"
        "  [2] Abort\n"
        "  [3] Continue with feedback\n"
        "Choice [1/2/3]:"
    ).strip()
    if answer in ("2", "abort"):
        return "abort", ""
    if answer in ("3", "feedback", "3 continue with feedback"):
        text = prompt_fn("Feedback:").strip()
        return "feedback", text
    return "continue", ""


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def get_standard_tools(session) -> tuple[list, dict]:
    """Return (tools_list, tool_handlers) bound to `session` for use with rollout().

    File/prompt tools mutate the session (backups, prompt backend); shell tools share its job
    registry; web tools are stateless. A fresh session per agent isolates all of this, so there
    is no per-rollout reset to do here anymore.
    """
    def write_file(path: str, content: str, append: bool = False) -> str:
        """Write or append content to a file, creating parent directories as needed.
        Args:
            path: Absolute or relative path to the target file.
            content: The text content to write.
            append: If true, append to existing file instead of overwriting.
        """
        return _write_file(session, path, content, append)

    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace an exact substring in a file. View the file first (e.g. `cat`/`Get-Content`) to get exact text.
        Args:
            path: File to edit.
            old: Exact text to find (must be unique in the file unless replace_all is set).
            new: Replacement text.
            replace_all: If true, replace every occurrence instead of requiring uniqueness.
        """
        return _edit_file(session, path, old, new, replace_all)

    def prompt_user(question: str) -> str:
        """Ask the user a question and return their response.
        Args:
            question: The question or message to display to the user.
        """
        return session.prompt_backend(question)

    tools = (shell_tools.make_shell_tools(session)
             + [write_file, edit_file, web_search, web_fetch, prompt_user])
    handlers = {fn.__name__: fn for fn in tools}
    return tools, handlers
