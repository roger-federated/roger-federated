"""Standard tools for the agent rollout loop.

Each tool is a plain callable with Google-style docstrings so that HF's
apply_chat_template(tools=...) can auto-generate JSON schemas from the
signature + docstring.  All tools return str and never raise — errors
become descriptive strings the agent can reason about and retry.

Usage:
    tools, handlers = get_standard_tools()
    await rollout(..., tools=tools, tool_handlers=handlers)
"""

import subprocess, os, re, pathlib, shutil, time, ast, math, asyncio


# ---------------------------------------------------------------------------
# Calculator internals
# ---------------------------------------------------------------------------

# Expose all public math symbols plus a few builtins
_MATH_NS: dict = {k: v for k, v in vars(math).items() if not k.startswith("_")}
_MATH_NS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})

# AST node whitelist — anything not in here is rejected before eval
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

# Timer registry — id ("t1", "t2", ...) -> {"fires_at": float, "label": str}
# fires_at = time.monotonic() + seconds; pure timestamp math, no background tasks.
_timers: dict[str, dict] = {}
_timer_seq: int = 0

# Stopwatch registry — id ("w1", "w2", ...) -> {"start_at": float, "stopped_at": float|None, "label": str}
_stopwatches: dict[str, dict] = {}
_stopwatch_seq: int = 0

# Background command jobs — id ("c1", ...) -> {"proc": Popen, "future": Future, "command": str, "reported": bool}
# Process is created eagerly (so stop_command gets a handle); only the blocking wait runs in a worker thread.
_jobs: dict[str, dict] = {}
_job_seq: int = 0


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

def run_command(command: str, timeout: int = 30, background: bool = False) -> str:
    """Execute a shell command and return its output.
    Args:
        command: The shell command to execute (supports pipes, redirects, chaining).
        timeout: Max seconds before the process is killed; 0 means no limit.
        background: If true, run the command in the background and return a job id
            immediately; its output is delivered automatically once it finishes.
            Use check_command to poll and stop_command to kill it.
    """
    global _policy, _job_seq
    # Lazy-load policy on first call, re-read each time to pick up user edits
    _policy = _load_policy(_policy_file)

    status = _check_policy(command, _policy)
    if status == "blocked":
        return f"Blocked by policy: '{command}' matches a blocked command pattern in {_policy_file}"
    if status == "confirm":
        answer = _prompt_backend(f"Agent wants to run: {command}\nAllow? [y/n]")
        if answer.strip().lower() not in ("y", "yes"):
            return f"Command rejected by user: {command}"

    if background:
        # Spawn now (instant) so stop_command has a handle; wait in a worker thread.
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        fut = asyncio.get_running_loop().run_in_executor(None, _await_proc, proc, timeout)
        _job_seq += 1
        _jobs[f"c{_job_seq}"] = {"proc": proc, "future": fut, "command": command, "reported": False}
        return f"command c{_job_seq} started in background (timeout {timeout}s)"

    try:
        return _await_proc(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True), timeout)
    except Exception as e:
        return f"Error: {e}"


def _await_proc(proc: subprocess.Popen, timeout: int) -> str:
    """Block until proc finishes (or timeout), returning exit code + combined output."""
    try:
        out, err = proc.communicate(timeout=timeout or None)
        return (f"exit {proc.returncode}\n" + (out or "") + (err or "")).rstrip()
    except subprocess.TimeoutExpired:
        proc.kill(); proc.communicate()
        return f"Command timed out after {timeout}s"


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
        max_results: Maximum number of matching paths to return.
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
    # Normalize alternative notations to Python syntax
    expr = (expression
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")   # Unicode minus −
        .replace("²", "**2") # superscript ²
        .replace("³", "**3") # superscript ³
        .strip()
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
        # Clean up floats that are exact integers (e.g. sqrt(4) → 2 not 2.0)
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.10g}"
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"


def prompt_user(question: str) -> str:
    """Ask the user a question and return their response.
    Args:
        question: The question or message to display to the user.
    """
    return _prompt_backend(question)


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
        lines = []
        now = time.monotonic()
        for tid, t in _timers.items():
            rem = t["fires_at"] - now
            sfx = f" ({t['label']})" if t["label"] else ""
            if rem > 0:
                lines.append(f"{tid} pending, {rem:.1f}s remaining{sfx}")
            else:
                lines.append(f"{tid} fired {-rem:.1f}s ago{sfx}")
        return "\n".join(lines)

    if timer_id not in _timers:
        return f"unknown timer id: {timer_id}"

    t = _timers[timer_id]
    sfx = f" ({t['label']})" if t["label"] else ""
    rem = t["fires_at"] - time.monotonic()
    if rem > 0:
        if wait:
            await asyncio.sleep(rem)  # yields the event loop; non-blocking
        else:
            return f"{timer_id} pending, {rem:.1f}s remaining{sfx}"
    # Re-check elapsed after optional sleep
    ago = time.monotonic() - t["fires_at"]
    return f"{timer_id} fired {ago:.1f}s ago{sfx}"


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
        lines = []
        now = time.monotonic()
        for wid, w in _stopwatches.items():
            elapsed = (w["stopped_at"] if w["stopped_at"] is not None else now) - w["start_at"]
            sfx = f" ({w['label']})" if w["label"] else ""
            if w["stopped_at"] is not None:
                lines.append(f"{wid} stopped at {elapsed:.2f}s{sfx}")
            else:
                lines.append(f"{wid} running, {elapsed:.2f}s elapsed{sfx}")
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


def stop_command(job_id: str) -> str:
    """Force-stop a background command started by run_command.
    Args:
        job_id: Id returned by run_command(background=True), e.g. 'c1'.
    """
    job = _jobs.get(job_id)
    if not job:
        return f"unknown command id: {job_id}"
    if job["future"].done():
        return f"{job_id} already finished"
    job["proc"].kill()  # worker's communicate() returns, completing the future
    return f"{job_id} stopped"


def check_command(job_id: str = "") -> str:
    """Check a background command's status/output, or list all background commands.
    Args:
        job_id: Id from run_command(background=True). Empty string lists all jobs.
    """
    if not job_id:
        if not _jobs:
            return "no background commands"
        return "\n".join(f"{jid} {'finished' if j['future'].done() else 'running'}: {j['command']}"
                         for jid, j in _jobs.items())
    job = _jobs.get(job_id)
    if not job:
        return f"unknown command id: {job_id}"
    if not job["future"].done():
        return f"{job_id} still running" # TODO: should include output thus far
    return f"{job_id} finished:\n{job['future'].result()}"


# ---------------------------------------------------------------------------
# Background-job helpers (not agent tools; called directly by the rollout)
# ---------------------------------------------------------------------------

def drain_finished_jobs() -> list[tuple[str, str]]:
    """Return (id, output) for jobs that finished since the last drain, marking them reported."""
    out = []
    for jid, job in _jobs.items():
        if job["future"].done() and not job["reported"]:
            job["reported"] = True
            out.append((jid, job["future"].result()))
    return out


def pending_jobs() -> list:
    """Futures of background commands still running (for the rollout to await)."""
    return [j["future"] for j in _jobs.values() if not j["future"].done()]


def terminate_jobs() -> None:
    """Kill any still-running background commands and clear the registry (rollout cleanup)."""
    for job in _jobs.values():
        if not job["future"].done():
            job["proc"].kill()
    _jobs.clear()


# ---------------------------------------------------------------------------
# End-of-rollout revert
# ---------------------------------------------------------------------------

def offer_revert(prompt_fn=None) -> tuple[bool, int]:
    """Prompt the user to revert files overwritten during the rollout.
    Returns (was_prompted, n_reverted): was_prompted is False when no backups
    existed (caller should then ask for an explicit outcome grade instead)."""
    if not _backups:
        return False, 0
    fn = prompt_fn or _prompt_backend

    listing = "\n".join(f"  {i+1}. {orig}" for i, (orig, _bak) in enumerate(_backups))
    answer = fn(f"The following files were overwritten and backed up:\n{listing}\nRevert? [all / 1,2,... / none]")
    answer = answer.strip().lower()

    if answer == "none" or not answer:
        _backups.clear()
        return True, 0

    if answer == "all":
        indices = list(range(len(_backups)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in answer.split(",")]
        except ValueError:
            _backups.clear()
            return True, 0

    n_reverted = 0
    for idx in indices:
        if 0 <= idx < len(_backups):
            orig, bak = _backups[idx]
            try:
                shutil.copy2(bak, orig)
                n_reverted += 1
            except OSError:
                pass
    _backups.clear()
    return True, n_reverted


# ---------------------------------------------------------------------------
# Max-steps check-in (not exposed as an agent tool; called directly by rollout)
# ---------------------------------------------------------------------------

def maxsteps_checkin(prompt_fn=None) -> tuple[str, str]:
    """Present the three-option max-steps check-in to the user.
    Returns (action, feedback_text) where action is one of:
        'continue'  — user is happy, loop continues          (small positive signal)
        'abort'     — user stops the rollout                 (large negative signal)
        'feedback'  — user continues but provides guidance   (small negative signal)
    feedback_text is non-empty only for the 'feedback' action.
    """
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
    # Default (empty / 1 / anything else) → continue
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
    global _prompt_backend, _policy_file, _timer_seq, _stopwatch_seq, _job_seq
    if prompt_backend is not None:
        _prompt_backend = prompt_backend
    _policy_file = policy_file
    # Reset per-rollout state so timers/stopwatches/jobs don't leak across episodes
    _timers.clear(); _timer_seq = 0
    _stopwatches.clear(); _stopwatch_seq = 0
    _jobs.clear(); _job_seq = 0

    tools = [run_command, read_file, write_file, search_file, search_dir, calculate, prompt_user,
             set_timer, check_timer, start_stopwatch, read_stopwatch, stop_command, check_command]
    handlers = {fn.__name__: fn for fn in tools}
    return tools, handlers
