"""Shell execution tools for the agent rollout loop.

Provides run_command (foreground + background), stop_command, check_command,
and the command-policy guardrail machinery.  Separated from std_tools to keep
concerns tidy; consumed by std_tools.get_standard_tools().

Call configure() once per rollout (done automatically via get_standard_tools)
to set the prompt backend, policy file, and reset per-episode job state.
"""

import subprocess, os, re, asyncio, tempfile

# ---------------------------------------------------------------------------
# Module-level per-rollout state
# ---------------------------------------------------------------------------

# Pluggable prompt backend — confirm prompts route through this
def _default_prompt(question: str) -> str:
    try:
        return input(question + " ")
    except EOFError:
        return "(no response — user input unavailable)"
    except KeyboardInterrupt:
        return "(user cancelled)"

_prompt_backend = _default_prompt

# Policy — path string + lazily-loaded dict (re-read each run_command call)
_policy_file: str = "command_policy.txt"
_policy: dict | None = None

# Background jobs: id ("c1", ...) -> {"proc": Popen, "future": Future,
# "command": str, "reported": bool, "outfile": str}
_jobs: dict[str, dict] = {}
_job_seq: int = 0


def configure(prompt_backend=None, policy_file: str = "command_policy.txt") -> None:
    """Set prompt backend + policy path, reset per-episode job state."""
    global _prompt_backend, _policy_file, _policy, _jobs, _job_seq
    if prompt_backend is not None:
        _prompt_backend = prompt_backend
    _policy_file = policy_file
    _policy = None          # force re-read on next run_command
    _jobs.clear()
    _job_seq = 0
    _ensure_scratch_ignored()


def _ensure_scratch_ignored() -> None:
    """Append .roger/ to .gitignore if cwd is a git repo and it is missing."""
    # Quick guard: skip when git is absent or cwd is not inside a repo
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode != 0:
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return

    gitignore = ".gitignore"
    try:
        existing = open(gitignore, encoding="utf-8").read() if os.path.isfile(gitignore) else ""
    except OSError:
        existing = ""

    # .* in gitignore already covers .roger/, so only add the explicit entry if .* is absent
    if ".roger/" in existing or ".*" in existing:
        return
    with open(gitignore, "a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(".roger/\n")
    print("[shell_tools] appended .roger/ to .gitignore")


# ---------------------------------------------------------------------------
# Policy internals
# ---------------------------------------------------------------------------

def _load_policy(path: str) -> dict[str, list[str]]:
    """Parse command_policy.txt → {"blocked": [...], "confirm": [...], "evasion": [...]}."""
    policy: dict[str, list[str]] = {"blocked": [], "confirm": [], "evasion": []}
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


def _is_protected(path: str) -> bool:
    """True when path resolves to the guardrail policy file."""
    try:
        a, b = os.path.abspath(path), os.path.abspath(_policy_file)
        return a.lower() == b.lower() if os.name == "nt" else a == b
    except (OSError, ValueError):
        return False


def _refs_protected(command: str) -> bool:
    """True when the command string contains the policy filename or its abspath.
    Broad by design — even reads of it via shell are blocked."""
    needle_base = os.path.basename(_policy_file).lower()
    needle_abs  = os.path.abspath(_policy_file).lower()
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
  Temp files:         use `.roger\\scratch\\` (gitignored); clean up when done.
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
  Temp files:         use `.roger/scratch/` (gitignored); clean up when done.
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

def run_command(command: str, timeout: int = 30, background: bool = False) -> str:
    """Execute a shell command and return its output. Uses PowerShell on Windows, /bin/sh elsewhere.
    Args:
        command: The shell command to execute (supports pipes, redirects, chaining).
        timeout: Max seconds before the process is killed; 0 means no limit.
        background: If true, run in the background and return a job id immediately.
            Use check_command to poll and stop_command to kill it.
    """
    global _policy, _job_seq
    _policy = _load_policy(_policy_file)   # re-read each call to pick up user edits

    if _refs_protected(command):
        return f"Blocked: command references the protected policy file {_policy_file}"

    status = _worst(_check_policy(command, _policy), _scan_scripts(command, _policy))
    if status == "blocked":
        return f"Blocked by policy: '{command}' matches a blocked command pattern in {_policy_file}"
    if status == "confirm":
        answer = _prompt_backend(f"Agent wants to run: {command}\nAllow? [y/n]")
        if answer.strip().lower() not in ("y", "yes"):
            return f"Command rejected by user: {command}"

    if background:
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".out").name
        proc = _popen(command, stdout=open(path, "w", encoding="utf-8"), stderr=subprocess.STDOUT)
        fut = asyncio.get_running_loop().run_in_executor(None, _await_bg, proc, timeout, path)
        _job_seq += 1
        _jobs[f"c{_job_seq}"] = {"proc": proc, "future": fut, "command": command,
                                  "reported": False, "outfile": path}
        return f"command c{_job_seq} started in background (timeout {timeout}s)"

    try:
        return _await_proc(_popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE), timeout)
    except Exception as e:
        return f"Error: {e}"


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
    job["proc"].kill()
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
        partial = _read_outfile(job["outfile"])
        return f"{job_id} still running" + (f", output so far:\n{partial}" if partial else "")
    return f"{job_id} finished:\n{job['future'].result()}"


# ---------------------------------------------------------------------------
# Background-job helpers (called directly by rollout; not agent-facing tools)
# ---------------------------------------------------------------------------

def drain_finished_jobs() -> list[tuple[str, str]]:
    """Return (id, output) for jobs that finished since last drain, marking them reported."""
    out = []
    for jid, job in _jobs.items():
        if job["future"].done() and not job["reported"]:
            job["reported"] = True
            out.append((jid, job["future"].result()))
    return out


def pending_jobs() -> list:
    """Futures of background commands still running."""
    return [j["future"] for j in _jobs.values() if not j["future"].done()]


def terminate_jobs() -> None:
    """Kill all still-running background commands and clear the registry (rollout cleanup)."""
    for job in _jobs.values():
        if not job["future"].done():
            job["proc"].kill()
        try: os.remove(job["outfile"])
        except OSError: pass
    _jobs.clear()
