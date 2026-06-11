"""Standard tools for the agent rollout loop.

Each tool is a plain callable with Google-style docstrings so that HF's
apply_chat_template(tools=...) can auto-generate JSON schemas from the
signature + docstring.  All tools return str and never raise — errors
become descriptive strings the agent can reason about and retry.

Usage:
    tools, handlers = get_standard_tools()
    await rollout(..., tools=tools, tool_handlers=handlers)
"""

import os, re, pathlib, shutil, time, ast, math, asyncio, tempfile
import shell_tools   # shell execution + policy machinery lives here
from path_utils import gutter


# ---------------------------------------------------------------------------
# Calculator internals
# ---------------------------------------------------------------------------

_MATH_NS: dict = {k: v for k, v in vars(math).items() if not k.startswith("_")}
_MATH_NS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})

_SAFE_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Name, ast.Call, ast.Load, ast.Tuple, ast.List,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE,
}

def _ast_safe(node: ast.AST) -> None:
    """Raise ValueError if the AST contains anything outside the math whitelist."""
    if type(node) not in _SAFE_NODES:
        raise ValueError(f"disallowed node: {type(node).__name__}")
    if isinstance(node, ast.Name) and node.id not in _MATH_NS:
        raise ValueError(f"unknown name: '{node.id}'")
    if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
        raise ValueError("only bare function calls allowed (e.g. sin(x), not obj.method())")
    for child in ast.iter_child_nodes(node):
        _ast_safe(child)


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

# Timer registry: id -> {"fires_at": float, "label": str}
_timers: dict[str, dict] = {}
_timer_seq: int = 0

# Stopwatch registry: id -> {"start_at": float, "stopped_at": float|None, "label": str}
_stopwatches: dict[str, dict] = {}
_stopwatch_seq: int = 0


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

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

    lo = (start_line - 1) if start_line > 0 else 0
    hi = end_line if end_line > 0 else len(lines)
    return gutter("".join(lines[lo:hi]), lo + 1)


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
    """Replace an exact substring in a file. Read the file first to get exact text.
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
    """Copy an existing file to .backups/{path}.{timestamp}.bak and record for revert."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    try:
        rel = os.path.relpath(path)
    except ValueError:
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

    matches = [f"{i}: {line.rstrip()}" for i, line in enumerate(lines, 1) if regex.search(line)]
    if not matches:
        return f"No matches found for '{pattern}' in {path}"
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
        max_results: Maximum number of matching paths to return.
    """
    root = pathlib.Path(path)
    if not root.is_dir():
        return f"Not a directory: {path}"
    results = []
    try:
        for p in root.rglob(pattern):
            rel = p.relative_to(root)
            if p.is_dir():
                results.append(f"{rel}/")
            else:
                try: size = p.stat().st_size
                except OSError: size = "?"
                results.append(f"{rel} ({size} bytes)")
            if len(results) >= max_results:
                break
    except OSError as e:
        return f"Error searching directory: {e}"
    if not results:
        return f"No files matching '{pattern}' in {path}"
    if len(results) >= max_results:
        results.append(f"... (truncated at {max_results} results)")
    return "\n".join(results)


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Args:
        expression: A math expression. Python operator syntax is used; additionally
            '^' means exponentiation, '×'/'÷' mean multiply/divide, '−' is a Unicode
            minus, and '²'/'³' expand to '**2'/'**3'. Scientific notation like 1.5e-3
            is natively supported. All math module functions and constants are available:
            sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh,
            sqrt, cbrt, exp, expm1, log, log2, log10, log1p,
            floor, ceil, trunc, factorial, gcd, lcm, comb, perm,
            hypot, degrees, radians, isnan, isinf,
            pi, e, tau, inf, nan.
            Examples: "2^10", "sqrt(2)*pi", "log(1000, 10)", "sin(pi/6)", "3.0e8 / 1e6"
    """
    expr = (expression
        .replace("^", "**").replace("×", "*").replace("÷", "/")
        .replace("−", "-").replace("²", "**2").replace("³", "**3").strip()
    )
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"Syntax error: {e}"
    try:
        _ast_safe(tree)
    except ValueError as e:
        return f"Unsafe expression — {e}"
    try:
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, _MATH_NS)
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.10g}"
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"


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
# Timers
# ---------------------------------------------------------------------------

def set_timer(seconds: float, label: str = "") -> str:
    """Set a countdown timer and return immediately with its id for later polling.
    Args:
        seconds: Duration in seconds before the timer fires.
        label: Optional description surfaced when the timer fires.
    """
    global _timer_seq
    _timer_seq += 1
    tid = f"t{_timer_seq}"
    _timers[tid] = {"fires_at": time.monotonic() + seconds, "label": label}
    return f"timer {tid} set for {seconds:g}s{f' ({label})' if label else ''}"


async def check_timer(timer_id: str = "", wait: bool = False) -> str:
    """Check the status of a timer, or list all timers.
    Args:
        timer_id: Id returned by set_timer (e.g. 't1'). Empty string lists all timers.
        wait: If true and the timer is still pending, suspend until it fires then return.
    """
    if not timer_id:
        if not _timers:
            return "no timers set"
        now = time.monotonic()
        lines = []
        for tid, t in _timers.items():
            rem = t["fires_at"] - now
            sfx = f" ({t['label']})" if t["label"] else ""
            lines.append(f"{tid} {'pending, ' + f'{rem:.1f}s remaining' if rem > 0 else 'fired ' + f'{-rem:.1f}s ago'}{sfx}")
        return "\n".join(lines)

    if timer_id not in _timers:
        return f"unknown timer id: {timer_id}"
    t = _timers[timer_id]
    sfx = f" ({t['label']})" if t["label"] else ""
    rem = t["fires_at"] - time.monotonic()
    if rem > 0:
        if wait:
            await asyncio.sleep(rem)
        else:
            return f"{timer_id} pending, {rem:.1f}s remaining{sfx}"
    ago = time.monotonic() - t["fires_at"]
    return f"{timer_id} fired {ago:.1f}s ago{sfx}"


# ---------------------------------------------------------------------------
# Stopwatches
# ---------------------------------------------------------------------------

def start_stopwatch(label: str = "") -> str:
    """Start a stopwatch and return its id for later reading.
    Args:
        label: Optional description surfaced when the stopwatch is read.
    """
    global _stopwatch_seq
    _stopwatch_seq += 1
    wid = f"w{_stopwatch_seq}"
    _stopwatches[wid] = {"start_at": time.monotonic(), "stopped_at": None, "label": label}
    return f"stopwatch {wid} started{f' ({label})' if label else ''}"


def read_stopwatch(stopwatch_id: str = "", stop: bool = False) -> str:
    """Read the elapsed time of a stopwatch, or list all stopwatches.
    Args:
        stopwatch_id: Id returned by start_stopwatch (e.g. 'w1'). Empty string lists all.
        stop: If true, freeze the stopwatch at the current elapsed time.
    """
    if not stopwatch_id:
        if not _stopwatches:
            return "no stopwatches started"
        now = time.monotonic()
        lines = []
        for wid, w in _stopwatches.items():
            elapsed = (w["stopped_at"] if w["stopped_at"] is not None else now) - w["start_at"]
            sfx = f" ({w['label']})" if w["label"] else ""
            state = f"stopped at {elapsed:.2f}s" if w["stopped_at"] is not None else f"running, {elapsed:.2f}s elapsed"
            lines.append(f"{wid} {state}{sfx}")
        return "\n".join(lines)

    if stopwatch_id not in _stopwatches:
        return f"unknown stopwatch id: {stopwatch_id}"
    w = _stopwatches[stopwatch_id]
    sfx = f" ({w['label']})" if w["label"] else ""
    now = time.monotonic()
    if stop and w["stopped_at"] is None:
        w["stopped_at"] = now
    elapsed = (w["stopped_at"] if w["stopped_at"] is not None else now) - w["start_at"]
    if w["stopped_at"] is not None:
        return f"{stopwatch_id} stopped at {elapsed:.2f}s{sfx}"
    return f"{stopwatch_id} running, {elapsed:.2f}s elapsed{sfx}"


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
    global _prompt_backend, _policy_file, _timer_seq, _stopwatch_seq
    if prompt_backend is not None:
        _prompt_backend = prompt_backend
    _policy_file = policy_file

    # Reset per-rollout state
    _timers.clear(); _timer_seq = 0
    _stopwatches.clear(); _stopwatch_seq = 0
    _backups.clear()

    # Configure shell_tools (sets its own prompt/policy globals and resets _jobs)
    shell_tools.configure(prompt_backend=prompt_backend, policy_file=policy_file)

    shell = [shell_tools.run_command, shell_tools.stop_command, shell_tools.check_command]
    file  = [read_file, write_file, edit_file, search_file, search_dir]
    misc  = [calculate, prompt_user, finish,
             set_timer, check_timer, start_stopwatch, read_stopwatch]
    tools = shell + file + misc
    handlers = {fn.__name__: fn for fn in tools}
    return tools, handlers
