"""Shell execution tools for the agent rollout loop.

Provides run_command (foreground + background), stop_command, check_command,
and the command-policy guardrail machinery.  Separated from std_tools to keep
concerns tidy; consumed by std_tools.get_standard_tools().

The agent-facing tools are session-bound: make_shell_tools(session) returns
run_command/stop_command/check_command closures that read/mutate the given
ToolSession (its job registry, prompt backend, and policy path), so concurrent
sub-agents never share a job registry.  The policy parsing helpers below are
stateless (they take the policy dict / policy_file explicitly).
"""

import subprocess, os, re, asyncio, tempfile


# ---------------------------------------------------------------------------
# Policy internals
# ---------------------------------------------------------------------------

def _parse_policy(lines) -> dict[str, list[str]]:
    """Parse [section] / pattern lines → {"blocked": [...], "confirm": [...], "evasion": [...]}."""
    policy: dict[str, list[str]] = {"blocked": [], "confirm": [], "evasion": []}
    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            if section not in policy:
                policy[section] = []
        elif section:
            policy[section].append(line)
    return policy


def _load_policy(path: str) -> dict[str, list[str]]:
    """Load the command policy: a cwd override file if present, else the packaged default."""
    if os.path.isfile(path):                         # project-local override in cwd
        with open(path, encoding="utf-8") as f:
            return _parse_policy(f)
    # Shipped default — resolve as package data so guardrails work after install anywhere
    from importlib.resources import files
    text = files("roger.tools").joinpath("command_policy.txt").read_text(encoding="utf-8")
    return _parse_policy(text.splitlines())


def _is_protected(path: str, policy_file: str = "command_policy.txt") -> bool:
    """True when path resolves to the guardrail policy file."""
    try:
        a, b = os.path.abspath(path), os.path.abspath(policy_file)
        return a.lower() == b.lower() if os.name == "nt" else a == b
    except (OSError, ValueError):
        return False


def _refs_protected(command: str, policy_file: str = "command_policy.txt") -> bool:
    """True when the command string contains the policy filename or its abspath.
    Broad by design — even reads of it via shell are blocked."""
    needle_base = os.path.basename(policy_file).lower()
    needle_abs  = os.path.abspath(policy_file).lower()
    cmd_lower   = command.lower()
    return needle_base in cmd_lower or needle_abs in cmd_lower


def _worst(a: str, b: str) -> str:
    """Return the stricter of two policy verdicts: blocked > confirm > allowed."""
    _rank = {"blocked": 2, "confirm": 1, "allowed": 0}
    return a if _rank.get(a, 0) >= _rank.get(b, 0) else b


def _check_policy(command: str, policy: dict[str, list[str]]) -> str:
    """Split on shell operators and check each segment.
    Returns 'blocked', 'confirm', or 'allowed'."""
    segments = re.split(r"\|{1,2}|&&|;|\n", command)
    verdict = "allowed"
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        for rule in policy.get("blocked", []):
            if seg.startswith(rule) or seg == rule:
                return "blocked"
        for rule in policy.get("confirm", []):
            if seg.startswith(rule) or seg == rule:
                verdict = "confirm"
        # Evasion entries: substring match (prefix matching misses mid-command tokens)
        for token in policy.get("evasion", []):
            if token.lower() in seg.lower():
                verdict = _worst(verdict, "confirm")
    return verdict


def _scan_scripts(command: str, policy: dict) -> str:
    """Scan any script files referenced in command through _check_policy line-by-line."""
    _MAX_BYTES = 64 * 1024
    _SCRIPT_RE = re.compile(r"""[^\s'";&|]+\.(?:bat|cmd|ps1|sh|vbs|py)\b""", re.I)
    verdict = "allowed"
    for m in _SCRIPT_RE.finditer(command):
        path = m.group(0).strip("'\"")
        try:
            if os.path.getsize(path) > _MAX_BYTES:
                verdict = _worst(verdict, "confirm"); continue
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(("#", "::")):
                        continue
                    verdict = _worst(verdict, _check_policy(line, policy))
                    if verdict == "blocked":
                        return "blocked"
        except FileNotFoundError:
            pass
        except OSError:
            verdict = _worst(verdict, "confirm")
    return verdict


# ---------------------------------------------------------------------------
# System-prompt idioms block
# ---------------------------------------------------------------------------

def shell_idioms() -> str:
    """Return an OS-specific shell cheat-sheet for injection into the system prompt."""
    if os.name == "nt":
        return """\
Shell idioms for common commands (PowerShell via run_command):
  Read file:          `Get-Content f`  (alias: `gc f`)
  Read lines N-M:     `(gc f)[N-1..M-1]`          # 0-indexed slices
  Locate (+ lines):   `Select-String -n 'pat' f`
  Find files:         `Get-ChildItem -Recurse -Filter *.py`
  Copy region A→B:    `(gc a)[N-1..M-1] | Add-Content b`     # no re-typing
  Append to file:     `"text" | Add-Content f`
  Calculate:          `[math]::Sqrt(2) * [math]::Pi`   or   `(2 + 3) * 4`
  Time a command:     `Measure-Command { run_command "..." }`
  Delay:              `Start-Sleep -Seconds N`
  Temp files:         use `~\\.roger\\scratch\\` (outside the project); clean up when done.
  Edit a file:        edit_file(path, old, new) — consider copying `old` verbatim from Get-Content so it matches
                      exactly; to insert without removing, set `new = old + your_addition`.
Emit *new* file content with write_file/edit_file, not PowerShell here-strings."""
    else:
        return """\
Shell idioms for common commands (sh via run_command):
  Read file:          `cat f`
  Read lines N-M:     `sed -n 'N,Mp' f`
  Locate (+ lines):   `grep -rn 'pat' .`
  Find files:         `find . -name '*.py'`
  Copy region A→B:    `sed -n 'N,Mp' a >> b`          # no re-typing
  Append to file:     `echo "text" >> f`
  Calculate:          `awk 'BEGIN{print sqrt(2)*atan2(0,-1)}'`   or   `bc -l <<<'2^10'`
  Time a command:     `time <cmd>`
  Delay:              `sleep N`
  Temp files:         use `~/.roger/scratch/` (outside the project); clean up when done.
  Edit a file:        edit_file(path, old, new) — consider copying `old` verbatim from cat so it matches
                      exactly; to insert without removing, set `new = old + your_addition`.
Emit *new* file content with write_file/edit_file, not heredocs."""


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _popen(command: str, **kw) -> subprocess.Popen:
    """Spawn command through the platform shell (PowerShell on Windows, /bin/sh elsewhere)."""
    if os.name == "nt":
        return subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            text=True, **kw)
    return subprocess.Popen(command, shell=True, text=True, **kw)


def _await_proc(proc: subprocess.Popen, timeout: int) -> str:
    """Block until proc finishes (or timeout); return exit code + combined output."""
    try:
        out, err = proc.communicate(timeout=timeout or None)
        return (f"exit {proc.returncode}\n" + (out or "") + (err or "")).rstrip()
    except subprocess.TimeoutExpired:
        proc.kill(); proc.communicate()
        return f"Command timed out after {timeout}s"


def _read_outfile(path: str) -> str:
    """Current contents of a background job's tee file ('' if unreadable)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _await_bg(proc: subprocess.Popen, timeout: int, path: str) -> str:
    """Wait on a background proc (output tee'd to path); return exit code + output."""
    try:
        proc.wait(timeout=timeout or None)
        return (f"exit {proc.returncode}\n" + _read_outfile(path)).rstrip()
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait()
        return (f"Command timed out after {timeout}s\n" + _read_outfile(path)).rstrip()


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------

def _run_command(session, command: str, timeout: int, background: bool) -> str:
    policy = _load_policy(session.policy_file)   # re-read each call to pick up user edits

    if _refs_protected(command, session.policy_file):
        return f"Blocked: command references the protected policy file {session.policy_file}"

    status = _worst(_check_policy(command, policy), _scan_scripts(command, policy))
    if status == "blocked":
        return f"Blocked by policy: '{command}' matches a blocked command pattern in {session.policy_file}"
    if status == "confirm":
        answer = session.prompt_backend(f"Agent wants to run: {command}\nAllow? [y/n]")
        if answer.strip().lower() not in ("y", "yes"):
            return f"Command rejected by user: {command}"

    if background:
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".out").name
        proc = _popen(command, stdout=open(path, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
        fut = asyncio.get_running_loop().run_in_executor(None, _await_bg, proc, timeout, path)
        session.job_seq += 1
        session.jobs[f"c{session.job_seq}"] = {"proc": proc, "future": fut, "command": command,
                                               "reported": False, "outfile": path}
        return f"command c{session.job_seq} started in background (timeout {timeout}s)"

    try:
        return _await_proc(_popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE), timeout)
    except Exception as e:
        return f"Error: {e}"


def _stop_command(session, job_id: str) -> str:
    job = session.jobs.get(job_id)
    if not job:
        return f"unknown command id: {job_id}"
    if job["future"].done():
        return f"{job_id} already finished"
    job["proc"].kill()
    return f"{job_id} stopped"


def _check_command(session, job_id: str) -> str:
    if not job_id:
        if not session.jobs:
            return "no background commands"
        return "\n".join(f"{jid} {'finished' if j['future'].done() else 'running'}: {j['command']}"
                         for jid, j in session.jobs.items())
    job = session.jobs.get(job_id)
    if not job:
        return f"unknown command id: {job_id}"
    if not job["future"].done():
        partial = _read_outfile(job["outfile"])
        return f"{job_id} still running" + (f", output so far:\n{partial}" if partial else "")
    return f"{job_id} finished:\n{job['future'].result()}"


def make_shell_tools(session) -> list:
    """Return [run_command, stop_command, check_command] bound to `session` (its job registry,
    prompt backend, and policy path). The closures keep the exact signatures/docstrings the model
    sees (schema generation is unaffected) and delegate to the session-taking impls above."""
    def run_command(command: str, timeout: int = 30, background: bool = False) -> str:
        """Execute a shell command and return its output. Uses PowerShell on Windows, /bin/sh elsewhere.
        Args:
            command: The shell command to execute (supports pipes, redirects, chaining).
            timeout: Max seconds before the process is killed; 0 means no limit.
            background: If true, run in the background and return a job id immediately.
                Use check_command to poll and stop_command to kill it.
        """
        return _run_command(session, command, timeout, background)

    def stop_command(job_id: str) -> str:
        """Force-stop a background command started by run_command.
        Args:
            job_id: Id returned by run_command(background=True), e.g. 'c1'.
        """
        return _stop_command(session, job_id)

    def check_command(job_id: str = "") -> str:
        """Check a background command's status/output, or list all background commands.
        Args:
            job_id: Id from run_command(background=True). Empty string lists all jobs.
        """
        return _check_command(session, job_id)

    return [run_command, stop_command, check_command]
