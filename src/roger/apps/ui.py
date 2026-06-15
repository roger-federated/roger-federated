"""ui.py — Rich + prompt_toolkit terminal UI for the Roger CLI.

Public API:
  StreamRenderer(verbose, think_delims, tool_delims) — on_token callback; collapses thinking-channel blocks
  render_tool_call(name, args)     — pretty-print a tool invocation
  render_tool_result(name, result) — pretty-print a tool result
  make_prompt_backend()            — prompt_toolkit-backed input() replacement
  select_root(default)             — interactive folder selection (tkinter or text)
  read_prompt(session)             — multi-line prompt_toolkit input
"""
import os, sys, difflib, re, threading
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.syntax import Syntax
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle

from roger.agency.path_utils import state_dir

# ---------------------------------------------------------------------------
# Shared console (stderr=False keeps stdout clean for potential piping)
# ---------------------------------------------------------------------------
console = Console(highlight=False)


# ---------------------------------------------------------------------------
# StreamRenderer — state machine for one agent turn
# ---------------------------------------------------------------------------

class StreamRenderer:
    """Feed text chunks from on_token; collapses thinking-channel blocks inline.

    think_delims: (open_str, close_str) scraped from the tokenizer by
                  model_setup.find_think_delims — e.g. ("<|channel>", "<channel|>")
                  for Gemma-4. None → skip thinking detection (stream raw).
    tool_delims:  (open_str, close_str) decoded from tool_tokens — e.g.
                  ("<|tool_call>", "<tool_call|>"). None → skip suppression.
    Stateful because a delimiter can arrive split across chunks.
    One instance per rollout turn; do not reuse.
    """

    def __init__(self, verbose: bool = False,
                 think_delims: tuple[str, str] | None = None,
                 tool_delims:  tuple[str, str] | None = None):
        self._verbose     = verbose
        self._think_open  = think_delims[0] if think_delims else None
        self._think_close = think_delims[1] if think_delims else None
        self._tool_open   = tool_delims[0]  if tool_delims  else None
        self._tool_close  = tool_delims[1]  if tool_delims  else None
        # States: "answer" | "thinking" | "tool"
        self._state       = "answer"
        self._buf         = ""          # cross-chunk accumulator
        self._think_start = None
        import time; self._time = time

    def feed(self, chunk: str) -> None:
        self._buf += chunk
        self._flush()

    def _flush(self) -> None:
        """Process buf until no more complete transitions can be resolved."""
        while True:
            if self._state == "answer":
                # Candidate transition points; pick the earliest
                t_idx = self._buf.find(self._think_open) if self._think_open else -1
                c_idx = self._buf.find(self._tool_open)  if self._tool_open  else -1
                # Earliest positive hit
                if t_idx == -1 and c_idx == -1:
                    # No delimiter starting — safe to flush up to a potential partial at tail.
                    # Guard length = longest possible delimiter prefix we might be mid-stream on.
                    guard = max((len(self._think_open) if self._think_open else 0),
                                (len(self._tool_open)  if self._tool_open  else 0)) - 1
                    safe  = self._buf[:max(0, len(self._buf) - guard)]
                    if safe:
                        console.print(safe, end="", markup=False)
                        self._buf = self._buf[len(safe):]
                    return
                # Determine which delimiter fires first
                if   t_idx == -1: first, state = c_idx, "tool"
                elif c_idx == -1: first, state = t_idx, "thinking"
                else:             first, state = (t_idx, "thinking") if t_idx <= c_idx else (c_idx, "tool")
                # Flush text before the delimiter
                if first > 0:
                    console.print(self._buf[:first], end="", markup=False)
                delim = self._think_open if state == "thinking" else self._tool_open
                self._buf   = self._buf[first + len(delim):]
                self._state = state
                if state == "thinking":
                    self._think_start = self._time.monotonic()
                    if not self._verbose:
                        console.print("\n[dim]● Thinking…[/dim]", end="", markup=True)
                continue

            elif self._state == "thinking":
                idx = self._buf.find(self._think_close)
                if idx == -1:
                    # Still inside — print if verbose, else discard
                    guard = len(self._think_close) - 1
                    safe  = self._buf[:max(0, len(self._buf) - guard)]
                    if self._verbose and safe:
                        console.print(safe, end="", markup=False,
                                      style=Style(color="grey50", italic=True))
                    if safe: self._buf = self._buf[len(safe):]
                    return
                elapsed      = self._time.monotonic() - self._think_start
                thinking_txt = self._buf[:idx]
                self._buf    = self._buf[idx + len(self._think_close):]
                self._state  = "answer"
                if self._verbose:
                    if thinking_txt:
                        console.print(thinking_txt, end="", markup=False,
                                      style=Style(color="grey50", italic=True))
                    console.print(f"\n[dim]● Thought for {elapsed:.1f}s[/dim]\n", markup=True)
                else:
                    console.print(f"\r[dim]● Thought for {elapsed:.1f}s[/dim]   \n", markup=True)
                continue

            elif self._state == "tool":
                # Suppress until the close delimiter; discard the whole span.
                idx = self._buf.find(self._tool_close) if self._tool_close else -1
                if idx != -1:
                    self._buf   = self._buf[idx + len(self._tool_close):]
                    self._state = "answer"
                    continue
                return  # close not yet arrived; wait for more chunks


# ---------------------------------------------------------------------------
# Tool call / result panels
# ---------------------------------------------------------------------------

_TOOL_ICONS = {
    "run_command": "⚡", "stop_command": "⛔", "check_command": "🔍",
    "read_file": "📄", "write_file": "✏️",  "edit_file": "✏️",
    "search_file": "🔎", "search_dir": "🗂️",
    "calculate": "🧮", "prompt_user": "💬", "finish": "✅",
    "load_tools": "🔧", "load_skill": "📚",
}
_DEFAULT_ICON = "⚙"


# ---------------------------------------------------------------------------
# Shell command classifier — maps idiom commands to (icon, past_label, detail)
# ---------------------------------------------------------------------------

# Each entry: (regex matching the command verb/prefix, icon, past_label, detail_fn)
# detail_fn receives the remainder of the command string after the verb.
_CMD_PATTERNS: list[tuple] = [
    # read: Get-Content/gc/cat/type — extract first non-flag token as filename
    (re.compile(r"^\s*(?:Get-Content|gc|cat|type)\s+", re.I),
     "📄", "Read",
     lambda rest: re.split(r"\s+", rest.strip())[0] if rest.strip() else "…"),
    # locate/search: Select-String/grep — extract pattern then file
    (re.compile(r"^\s*(?:Select-String|grep)\b", re.I),
     "🔍", "Searched for",
     lambda rest: " ".join(rest.split()[:2]) if rest.strip() else "…"),
    # find files: Get-ChildItem/gci/find
    (re.compile(r"^\s*(?:Get-ChildItem|gci|find)\b", re.I),
     "🗂️", "Found files",
     lambda rest: rest.strip()[:60] or "…"),
    # copy/append: Add-Content/>>
    (re.compile(r"^\s*(?:Add-Content|.*>>\s*\S)", re.I),
     "📋", "Copied",
     lambda rest: rest.strip()[:60] or "…"),
]

def _classify_command(command: str):
    """Return (icon, past_label, detail) if command matches a known shell idiom, else None."""
    for pat, icon, label, detail_fn in _CMD_PATTERNS:
        m = pat.match(command)
        if m:
            return icon, label, detail_fn(command[m.end():])
    return None


def _render_diff(path: str, old: str, new: str) -> None:
    """Render a colored unified diff for an edit_file result."""
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=path, tofile=path, lineterm=""))
    if not lines:
        return  # no textual change
    diff_text = "\n".join(lines)
    title = Text(f"✏️  edit_file — {path}", style="bold green")
    console.print(Panel(Syntax(diff_text, "diff", background_color="default"),
                        title=title, border_style="green", expand=False, padding=(0, 1)))


def render_tool_call(name: str, args: dict) -> None:
    """Print a formatted panel for an outgoing tool call.

    run_command calls that match a known shell idiom are suppressed here —
    the past-tense summary is emitted after completion in render_tool_result.
    """
    if name == "run_command" and not args.get("background") and _classify_command(args.get("command", "")):
        return  # intent line shown post-completion
    icon  = _TOOL_ICONS.get(name, _DEFAULT_ICON)
    title = Text(f"{icon} {name}", style="bold cyan")
    lines = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:200] + "…"
        lines.append(f"[bold]{k}[/bold] = {v_str}")
    body = "\n".join(lines) if lines else "[dim](no arguments)[/dim]"
    console.print(Panel(body, title=title, border_style="cyan", expand=False,
                        padding=(0, 1)))


def render_tool_result(name: str, result, args: dict = None) -> None:
    """Print a formatted panel (or concise intent line) for a tool result."""
    # --- edit_file diff ---
    if name == "edit_file" and args and str(result).startswith("Replaced"):
        _render_diff(args.get("path", "?"), args.get("old", ""), args.get("new", ""))
        return

    # --- shell intent lines (foreground only; background returns a job-id string, not output) ---
    if name == "run_command" and args and not args.get("background"):
        cls = _classify_command(args.get("command", ""))
        if cls:
            icon, label, detail = cls
            result_str = str(result)
            failed = bool(re.match(r"exit [^0]", result_str))  # non-zero exit
            if failed:
                # Show intent + raw output on failure so the error is visible
                console.print(f"[bold red]{icon} {label}:[/bold red] [red]{detail}[/red]")
                _print_raw_result(name, result_str)
            else:
                suffix = ""
                if label == "Read":  # append line count for reads, suppress content
                    n = result_str.count("\n")
                    suffix = f" [dim]— {n} line{'s' if n != 1 else ''}[/dim]"
                console.print(f"[bold]{icon} {label}:[/bold] {detail}{suffix}")
            return

    # --- generic fallback ---
    _print_raw_result(name, result)


def _print_raw_result(name: str, result) -> None:
    """Generic green panel for a tool result (shared fallback)."""
    if isinstance(result, (bytes, bytearray)):
        text = f"<{len(result)} bytes>"
    else:
        text = str(result)
    if len(text) > 2000:
        text = text[:2000] + f"\n[dim]…({len(text) - 2000} chars truncated)[/dim]"
    title = Text(f"↩ {name}", style="bold green")
    console.print(Panel(text, title=title, border_style="green", expand=False,
                        padding=(0, 1)))


# ---------------------------------------------------------------------------
# Prompt backend — replaces blocking input() for agent-facing prompts
# ---------------------------------------------------------------------------

def make_prompt_backend(session: PromptSession) -> Callable[[str], str]:
    """Return an input()-compatible callable backed by prompt_toolkit.

    Tool handlers run on the asyncio loop thread, where prompt_toolkit's sync .prompt()
    would call asyncio.run() and fail on the running loop. Run it in a worker thread that
    has no running loop instead, and block until it returns. The main prompt app is never
    running while a tool prompt fires, so reusing `session` across the thread is safe.
    """
    def _ask(question: str) -> str:
        box: dict = {}
        def worker():
            try:
                box["v"] = session.prompt(HTML(f"<ansiyellow>{question} </ansiyellow>"))
            except (EOFError, KeyboardInterrupt):
                box["v"] = "[user interrupted; no answer provided]"
        t = threading.Thread(target=worker)
        t.start(); t.join()
        return box["v"]
    return _ask


# ---------------------------------------------------------------------------
# Root folder selection
# ---------------------------------------------------------------------------

def select_root(default: str) -> str:
    """Ask the user to pick a project root folder.

    Tries tkinter filedialog first (shows a native folder picker); falls back
    to a text prompt if no display is available (headless/SSH).
    """
    console.print(
        f"\n[bold]Select the folder from which the agent will be working.[/bold]\n"
        f"Currently on: [cyan]{default}[/cyan]"
    )
    # Try native folder dialog
    path = None
    dialog_shown = False
    try:
        import tkinter as tk
        from tkinter import filedialog
        root_win = tk.Tk(); root_win.withdraw()
        picked = filedialog.askdirectory(initialdir=default,
                                         title="Select project root")
        root_win.destroy()
        dialog_shown = True
        path = picked or default  # Cancel returns "" → fall back to the default (cwd)
    except Exception:
        pass  # no display or tkinter missing — fall through to text input

    # Text prompt only when no GUI was available; a cancelled dialog already chose the default
    if path is None and not dialog_shown:
        # Text fallback with tab-completion
        try:
            from prompt_toolkit.completion import PathCompleter
            sess = PromptSession(completer=PathCompleter(only_directories=True,
                                                         expanduser=True))
            raw = sess.prompt(HTML(f"<ansiyellow>Root [{default}]: </ansiyellow>")).strip()
        except Exception:
            raw = input(f"Root [{default}]: ").strip()
        path = raw if raw else default

    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        console.print(f"[red]Directory not found:[/red] {path!r} — using default.")
        path = default
    console.print(f"[dim]Using root:[/dim] [cyan]{path}[/cyan]\n")
    return path


# ---------------------------------------------------------------------------
# Main prompt input
# ---------------------------------------------------------------------------

def make_session() -> PromptSession:
    """Create a prompt_toolkit session with persistent history."""
    history_path = os.path.join(state_dir(), "history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    return PromptSession(
        history=FileHistory(history_path),
        style=PTStyle.from_dict({"prompt": "bold cyan"}),
        multiline=False,   # shift-enter for multiline if desired later
    )


async def read_prompt(session: PromptSession) -> str | None:
    """Read a user prompt. Returns None only on EOF (Ctrl-D → quit); "" on empty input or Ctrl-C
    so the caller can re-prompt without quitting.

    Async because the REPL runs inside asyncio.run(_repl): prompt_toolkit's sync
    .prompt() would call asyncio.run() again and fail on the running loop. prompt_async
    is the supported in-loop entry point.
    """
    try:
        return (await session.prompt_async(HTML("<ansicyan><b>❯ </b></ansicyan>"))).strip()
    except EOFError:
        return None   # Ctrl-D → quit
    except KeyboardInterrupt:
        return ""     # Ctrl-C at prompt → treat as empty, re-prompt
