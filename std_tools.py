"""Standard tools for the agent rollout loop.

Each tool is a plain callable with Google-style docstrings so that HF's
apply_chat_template(tools=...) can auto-generate JSON schemas from the
signature + docstring.  All tools return str and never raise — errors
become descriptive strings the agent can reason about and retry.

Usage:
    tools, handlers = get_standard_tools()
    await rollout(..., tools=tools, tool_handlers=handlers)
"""

import subprocess, os, re, pathlib, shutil, time


# ---------------------------------------------------------------------------
# Module-level state (set by get_standard_tools, used by tools internally)
# ---------------------------------------------------------------------------

# Pluggable prompt backend — all user-facing prompts route through this
def _default_prompt(question: str) -> str:
    """Fallback prompt backend: blocking input() with EOFError/KeyboardInterrupt handling."""
    try:
        return input(question + " ")
    except EOFError:
        return "(no response — user input unavailable)"
    except KeyboardInterrupt:
        return "(user cancelled)"

_prompt_backend = _default_prompt

# Command policy — loaded lazily on first run_command call
_policy: dict | None = None
_policy_file: str = "command_policy.txt"

# Write backups — list of (original_path, backup_path) for end-of-rollout revert
_backups: list[tuple[str, str]] = []


# ---------------------------------------------------------------------------
# Command policy internals
# ---------------------------------------------------------------------------

def _load_policy(path: str) -> dict[str, list[str]]:
    """Parse command_policy.txt into {"blocked": [...], "confirm": [...]}.
    Missing file = empty lists (fully permissive). Lines starting with # are comments."""
    policy = {"blocked": [], "confirm": []}
    try:
        with open(path, encoding="utf-8") as f:
            section = None
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1]
                    if section not in policy:
                        policy[section] = []
                elif section:
                    policy[section].append(line)
    except FileNotFoundError:
        pass
    return policy


def _check_policy(command: str, policy: dict[str, list[str]]) -> str:
    """Check a command against the policy. Splits on shell operators to catch
    chained dangerous commands (e.g. 'echo x | sudo rm'). Returns
    "blocked", "confirm", or "allowed"."""
    # Split on pipe, &&, ||, ; to get individual segments
    segments = re.split(r"\|{1,2}|&&|;", command)
    verdict = "allowed"
    for seg in segments:
        seg = seg.strip()
        # Blocked rules take priority — check first
        for rule in policy.get("blocked", []):
            if seg.startswith(rule) or seg == rule:
                return "blocked"
        # Confirm rules escalate but don't short-circuit (a later segment may be blocked)
        for rule in policy.get("confirm", []):
            if seg.startswith(rule) or seg == rule:
                verdict = "confirm"
    return verdict


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def run_command(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return its output.
    Args:
        command: The shell command to execute (supports pipes, redirects, chaining).
        timeout: Maximum seconds to wait before killing the process.
    """
    global _policy
    # Lazy-load policy on first call, re-read each time to pick up user edits
    _policy = _load_policy(_policy_file)

    status = _check_policy(command, _policy)
    if status == "blocked":
        return f"Blocked by policy: '{command}' matches a blocked command pattern in {_policy_file}"
    if status == "confirm":
        answer = _prompt_backend(f"Agent wants to run: {command}\nAllow? [y/n]")
        if answer.strip().lower() not in ("y", "yes"):
            return f"Command rejected by user: {command}"

    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        # Combine stdout/stderr with exit code so the agent sees the full picture
        out = f"exit {r.returncode}\n"
        if r.stdout: out += r.stdout
        if r.stderr: out += r.stderr
        return out.rstrip()
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read the contents of a file, optionally a specific line range.
    Args:
        path: Absolute or relative path to the file.
        start_line: First line to read (1-indexed). 0 means start of file.
        end_line: Last line to read (1-indexed, inclusive). 0 means end of file.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except OSError as e:
        return f"Error reading file: {e}"

    # Slice to requested range (1-indexed, 0 = unset sentinel)
    lo = (start_line - 1) if start_line > 0 else 0
    hi = end_line if end_line > 0 else len(lines)
    selected = lines[lo:hi]

    # Prepend line numbers so the agent can reference specific locations
    return "".join(f"{lo + i + 1}: {line}" for i, line in enumerate(selected)).rstrip()


def write_file(path: str, content: str, append: bool = False) -> str:
    """Write or append content to a file, creating parent directories as needed.
    Args:
        path: Absolute or relative path to the target file.
        content: The text content to write.
        append: If true, append to existing file instead of overwriting.
    """
    try:
        # Back up existing file before overwriting (not on append, not on new file)
        if not append and os.path.isfile(path):
            _backup_file(path)

        # Auto-create parent dirs so the agent doesn't need a separate mkdir step
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


def _backup_file(path: str) -> None:
    """Copy an existing file to .backups/{path}.{timestamp}.bak and record it for revert."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Resolve to relative path if possible, keeping directory structure inside .backups/
    try:
        rel = os.path.relpath(path)
    except ValueError:
        # relpath fails across drives on Windows
        rel = os.path.basename(path)
    backup_path = os.path.join(".backups", f"{rel}.{ts}.bak")
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy2(path, backup_path)
    _backups.append((os.path.abspath(path), os.path.abspath(backup_path)))


def search_file(path: str, pattern: str, max_results: int = 50) -> str:
    """Search for a regex pattern in a file and return matching lines with line numbers.
    Args:
        path: Path to the file to search.
        pattern: Regular expression to match against each line.
        max_results: Maximum number of matching lines to return.
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return f"File not found: {path}"
    except PermissionError:
        return f"Permission denied: {path}"
    except OSError as e:
        return f"Error reading file: {e}"

    matches = []
    for i, line in enumerate(lines, 1):
        if regex.search(line):
            matches.append(f"{i}: {line.rstrip()}")

    if not matches:
        return f"No matches found for '{pattern}' in {path}"
    # Truncate to max_results to avoid flooding the context window
    total = len(matches)
    if total > max_results:
        matches = matches[:max_results]
        matches.append(f"... ({total} total matches, showing first {max_results})")
    return "\n".join(matches)


def search_dir(path: str, pattern: str, max_results: int = 100) -> str:
    """Find files matching a glob pattern within a directory tree.
    Args:
        path: Root directory to search from.
        pattern: Glob pattern to match (e.g. '*.py', '**/*.json').
    """
    root = pathlib.Path(path)
    if not root.is_dir():
        return f"Not a directory: {path}"

    results = []
    try:
        for p in root.rglob(pattern):
            # Show relative paths for readability; annotate with type/size
            rel = p.relative_to(root)
            if p.is_dir():
                results.append(f"{rel}/")
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = "?"
                results.append(f"{rel} ({size} bytes)")
            if len(results) >= max_results:
                break
    except OSError as e:
        return f"Error searching directory: {e}"

    if not results:
        return f"No files matching '{pattern}' in {path}"
    # Note truncation so the agent knows there may be more
    if len(results) >= max_results:
        results.append(f"... (truncated at {max_results} results)")
    return "\n".join(results)


def prompt_user(question: str) -> str:
    """Ask the user a question and return their response.
    Args:
        question: The question or message to display to the user.
    """
    return _prompt_backend(question)


# ---------------------------------------------------------------------------
# End-of-rollout revert
# ---------------------------------------------------------------------------

def offer_revert(prompt_fn=None) -> None:
    """Prompt the user to revert files overwritten during the rollout.
    Called at the end of rollout(). No-op if no backups were made."""
    if not _backups:
        return
    fn = prompt_fn or _prompt_backend

    listing = "\n".join(f"  {i+1}. {orig}" for i, (orig, _bak) in enumerate(_backups))
    answer = fn(f"The following files were overwritten and backed up:\n{listing}\nRevert? [all / 1,2,... / none]")
    answer = answer.strip().lower()

    if answer == "none" or not answer:
        _backups.clear()
        return

    # Determine which indices to revert
    if answer == "all":
        indices = list(range(len(_backups)))
    else:
        # Parse comma-separated numbers (1-indexed)
        try:
            indices = [int(x.strip()) - 1 for x in answer.split(",")]
        except ValueError:
            _backups.clear()
            return

    for idx in indices:
        if 0 <= idx < len(_backups):
            orig, bak = _backups[idx]
            try:
                shutil.copy2(bak, orig)
            except OSError:
                pass
    _backups.clear()


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

    tools = [run_command, read_file, write_file, search_file, search_dir, prompt_user]
    handlers = {fn.__name__: fn for fn in tools}
    return tools, handlers
