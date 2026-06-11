"""ui.py — Rich + prompt_toolkit terminal UI for the Roger CLI.

Public API:
  StreamRenderer(verbose)          — on_token callback that collapses thinking blocks
  render_tool_call(name, args)     — pretty-print a tool invocation
  render_tool_result(name, result) — pretty-print a tool result
  make_prompt_backend()            — prompt_toolkit-backed input() replacement
  select_root(default)             — interactive folder selection (tkinter or text)
  read_prompt(session)             — multi-line prompt_toolkit input
"""
import os, sys, re, textwrap
from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich import print as rprint
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle

# ---------------------------------------------------------------------------
# Shared console (stderr=False keeps stdout clean for potential piping)
# ---------------------------------------------------------------------------
console = Console(highlight=False)

# Thinking delimiters used by Gemma-4 / Gemma-3n and similar reasoning models.
# The model emits <think>…</think> around its reasoning block.
_THINK_OPEN  = "<think>"
_THINK_CLOSE = "</think>"

# Tool-call JSON is delimited by special tokens in the raw stream; we suppress
# it entirely from on_token output (already rendered via on_tool_call).
# We detect it by looking for the constrained-output prefix {"name":
_TOOL_JSON_RE = re.compile(r'\{"name":\s*"')


# ---------------------------------------------------------------------------
# StreamRenderer — state machine for one agent turn
# ---------------------------------------------------------------------------

class StreamRenderer:
    """Feed text chunks from on_token; collapses <think> blocks inline.

    Stateful because a delimiter can arrive split across chunks.
    One instance per rollout turn; do not reuse.
    """

    def __init__(self, verbose: bool = False):
        self._verbose = verbose
        # States: "answer" | "thinking" | "tool_json"
        self._state = "answer"
        self._buf   = ""           # cross-chunk accumulator
        self._think_secs = 0.0
        self._think_start = None
        self._think_line_printed = False
        import time; self._time = time

    def feed(self, chunk: str) -> None:
        self._buf += chunk
        self._flush()

    def _flush(self) -> None:
        """Process buf until no more complete transitions can be resolved."""
        while True:
            if self._state == "answer":
                # Check for start of thinking block
                idx = self._buf.find(_THINK_OPEN)
                if idx == -1:
                    # Check for start of tool JSON — suppress until we see matching close
                    m = _TOOL_JSON_RE.search(self._buf)
                    if m:
                        # Print everything before the JSON, then suppress
                        pre = self._buf[:m.start()]
                        if pre: console.print(pre, end="", markup=False)
                        self._buf = self._buf[m.start():]
                        self._state = "tool_json"
                        continue
                    # No special token starting — safe to print up to a potential
                    # partial delimiter at the end of the buffer
                    safe = self._buf[: len(self._buf) - len(_THINK_OPEN) + 1]
                    if safe:
                        console.print(safe, end="", markup=False)
                        self._buf = self._buf[len(safe):]
                    return
                # Print text before <think>
                if idx > 0:
                    console.print(self._buf[:idx], end="", markup=False)
                self._buf   = self._buf[idx + len(_THINK_OPEN):]
                self._state = "thinking"
                self._think_start = self._time.monotonic()
                if not self._verbose:
                    # Print collapsed placeholder; will be overwritten on close
                    console.print("\n[dim]● Thinking…[/dim]", end="", markup=True)
                    self._think_line_printed = True
                continue

            elif self._state == "thinking":
                idx = self._buf.find(_THINK_CLOSE)
                if idx == -1:
                    # Still inside thinking — print if verbose, else discard
                    safe = self._buf[: len(self._buf) - len(_THINK_CLOSE) + 1]
                    if self._verbose and safe:
                        console.print(safe, end="", markup=False,
                                      style=Style(color="grey50", italic=True))
                    if safe: self._buf = self._buf[len(safe):]
                    return
                elapsed = self._time.monotonic() - self._think_start
                thinking_text = self._buf[:idx]
                self._buf   = self._buf[idx + len(_THINK_CLOSE):]
                self._state = "answer"
                if self._verbose:
                    if thinking_text:
                        console.print(thinking_text, end="", markup=False,
                                      style=Style(color="grey50", italic=True))
                    console.print(f"\n[dim]● Thought for {elapsed:.1f}s[/dim]\n",
                                  markup=True)
                else:
                    # Overwrite placeholder with elapsed summary
                    console.print(f"\r[dim]● Thought for {elapsed:.1f}s[/dim]   \n",
                                  markup=True)
                continue

            elif self._state == "tool_json":
                # Suppress until the JSON object closes (balanced braces) then discard.
                # A simpler heuristic: the JSON ends when we exit the constrained window,
                # signalled by the matching close-tool token — but that's a token ID, not
                # text.  Instead we track brace depth; resets to answer when depth = 0.
                depth = 0
                end   = 0
                for i, ch in enumerate(self._buf):
                    if   ch == "{": depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end:
                    self._buf   = self._buf[end:]
                    self._state = "answer"
                    continue
                return  # JSON not complete yet; wait for more chunks


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
_DEFAULT_ICON = "🔧"


def render_tool_call(name: str, args: dict) -> None:
    """Print a formatted panel for an outgoing tool call."""
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


def render_tool_result(name: str, result) -> None:
    """Print a formatted panel for a tool result."""
    # Convert result to a displayable string; truncate long outputs
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
    """Return an input()-compatible callable backed by prompt_toolkit."""
    def _ask(question: str) -> str:
        try:
            return session.prompt(HTML(f"<ansiyellow>{question} </ansiyellow>"))
        except (EOFError, KeyboardInterrupt):
            return ""
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
        f"\n[bold]Project root[/bold] — the folder the agent will work in.\n"
        f"  Default: [cyan]{default}[/cyan]\n"
        f"  Press [bold]Enter[/bold] to accept, or type a path."
    )
    # Try native folder dialog
    path = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root_win = tk.Tk(); root_win.withdraw()
        picked = filedialog.askdirectory(initialdir=default,
                                         title="Select project root")
        root_win.destroy()
        if picked:
            path = picked
    except Exception:
        pass  # no display or tkinter missing — fall through to text input

    if path is None:
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
    history_path = os.path.join(os.path.expanduser("~"), ".roger", "history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    return PromptSession(
        history=FileHistory(history_path),
        style=PTStyle.from_dict({"prompt": "bold cyan"}),
        multiline=False,   # shift-enter for multiline if desired later
    )


def read_prompt(session: PromptSession) -> str | None:
    """Read a user prompt. Returns None on EOF (Ctrl-D → quit)."""
    try:
        text = session.prompt(HTML("<ansicyan><b>❯ </b></ansicyan>")).strip()
        return text if text else None
    except EOFError:
        return None   # Ctrl-D
    except KeyboardInterrupt:
        return ""     # Ctrl-C at prompt — caller can handle (abort current turn)
