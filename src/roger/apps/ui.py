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
from rich.live import Live
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import SimpleLexer
from prompt_toolkit.styles import Style as PTStyle

# Background highlight for the user's typed turn. SimpleLexer paints this style over every
# character of the buffer, so the tint hugs the typed text (not the whole line) and persists
# in the scrollback after submit — visually marking each user turn.
_USER_TURN_STYLE = "bg:#1e1e1e #f5f5f5"

from roger.agency.path_utils import state_dir

# ---------------------------------------------------------------------------
# Shared console (stderr=False keeps stdout clean for potential piping)
# ---------------------------------------------------------------------------
console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Compact LaTeX → Unicode (terminals can't render real math; approximate it)
# ---------------------------------------------------------------------------

_LATEX_SYMBOLS = {
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇", r"\sum": "∑", r"\prod": "∏",
    r"\int": "∫", r"\sqrt": "√", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\Rightarrow": "⇒", r"\to": "→", r"\in": "∈", r"\notin": "∉", r"\forall": "∀",
    r"\exists": "∃", r"\propto": "∝", r"\ldots": "…", r"\cdots": "⋯",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε",
    r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ", r"\pi": "π", r"\rho": "ρ",
    r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ", r"\Pi": "Π",
    r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}
_SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")

def _convert_math(m: str) -> str:
    for k, v in _LATEX_SYMBOLS.items():
        m = m.replace(k, v)
    m = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", m)
    m = re.sub(r"\^\{([^{}]*)\}", lambda g: g.group(1).translate(_SUP), m)
    m = re.sub(r"\^(.)", lambda g: g.group(1).translate(_SUP), m)
    m = re.sub(r"_\{([^{}]*)\}", lambda g: g.group(1).translate(_SUB), m)
    m = re.sub(r"_(.)", lambda g: g.group(1).translate(_SUB), m)
    m = re.sub(r"\\[a-zA-Z]+", "", m)          # drop leftover commands
    return m.replace("{", "").replace("}", "")

def _latex_to_unicode(text: str) -> str:
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: _convert_math(m.group(1)), text, flags=re.S)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: _convert_math(m.group(1)), text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: _convert_math(m.group(1)), text, flags=re.S)
    text = re.sub(r"\$(.+?)\$", lambda m: _convert_math(m.group(1)), text, flags=re.S)
    return text


# ---------------------------------------------------------------------------
# StreamRenderer — state machine for one agent turn
# ---------------------------------------------------------------------------

class StreamRenderer:
    """Feed text chunks from on_token; collapses thinking-channel blocks inline.

    think_delims: (open_str, close_str) decoded from model_setup.find_think_tokens —
                  e.g. ("<|channel>", "<channel|>") for Gemma-4.
                  None → skip thinking detection (stream raw).
    tool_delims:  (open_str, close_str) decoded from tool_tokens — e.g.
                  ("<|tool_call>", "<tool_call|>"). None → skip suppression.
    specials:     all decoded special-token strings; any that surface in answer text (turn close,
                  tool_response, eos, …) are dropped — the model emits them but they aren't content.
    Stateful because a delimiter can arrive split across chunks.
    Reused across the whole session. reset(suppress=) at each turn start, and flush() at turn
    end. This gives it an explicit per-turn lifecycle (see those methods).
    """

    def __init__(self, verbose: bool = False,
                 think_delims: tuple[str, str] | None = None,
                 tool_delims:  tuple[str, str] | None = None,
                 specials: list[str] | None = None):
        self._verbose     = verbose
        self._think_open  = think_delims[0] if think_delims else None
        self._think_close = think_delims[1] if think_delims else None
        self._tool_open   = tool_delims[0]  if tool_delims  else None
        self._tool_close  = tool_delims[1]  if tool_delims  else None
        # Answer-state delimiters: think/tool opens switch into a block; every *other* special
        # token (turn close, tool_response, eos, the close delims …) is structural noise the model
        # emits but should never be printed → dropped. specials = all decoded special-token strings.
        opens = {self._think_open, self._tool_open}
        drops = [s for s in dict.fromkeys(list(specials or []) + [self._think_close, self._tool_close])
                 if s and s not in opens]
        self._answer_delims = [(d, k) for d, k in
                               [(self._think_open, "thinking"), (self._tool_open, "tool")]
                               + [(s, "drop") for s in drops] if d]
        self._guard = max((len(d) for d, _ in self._answer_delims), default=1) - 1
        # States: "answer" | "thinking" | "tool"
        self._state       = "answer"
        self._buf         = ""          # cross-chunk accumulator
        self._answer      = ""          # visible answer text, re-rendered as live Markdown
        self._think_start = None
        self._status      = None        # live "Thinking…" spinner while collapsed
        self._live        = None        # live Markdown of the streaming answer
        import time; self._time = time

    def reset(self, suppress: bool = False) -> None:
        """Start a fresh turn. suppress=True begins in tool-suppression (memory turn: open delim seeded into the prompt, never streamed)."""
        self._stop_status()
        self._finalize_live()
        self._buf         = ""
        self._state       = "tool" if suppress else "answer"
        self._think_start = None
        if suppress:  # memory-write turn (Ctrl-D): show a spinner since its output is suppressed
            self._status = console.status("[dim]Updating memory…[/dim]", spinner="dots")
            self._status.start()

    def flush(self) -> None:
        """End of turn: stop the spinner, render the withheld answer tail, finalize the Markdown."""
        self._stop_status()
        if self._state == "answer" and self._buf:
            self._answer += self._buf
        self._buf = ""
        self._finalize_live()   # commits the rendered Markdown (its trailing newline frees the prompt)

    def _stop_status(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _render_live(self) -> None:
        md = Markdown(_latex_to_unicode(self._answer))
        if self._live is None:   # only one live display allowed at once; thinking spinner is stopped by now
            self._live = Live(md, console=console, refresh_per_second=8, vertical_overflow="visible")
            self._live.start()
        else:
            self._live.update(md)

    def _finalize_live(self) -> None:
        if self._live is not None:
            self._live.update(Markdown(_latex_to_unicode(self._answer)))
            self._live.stop()
            self._live = None
        self._answer = ""

    def feed(self, chunk: str) -> None:
        self._buf += chunk
        self._flush()

    def _flush(self) -> None:
        """Process buf until no more complete transitions can be resolved."""
        while True:
            if self._state == "answer":
                hits = [(p, d, k) for d, k in self._answer_delims
                        if (p := self._buf.find(d)) != -1]
                if not hits:
                    # No delimiter present — render up to a possible partial delimiter at the tail.
                    safe = self._buf[:max(0, len(self._buf) - self._guard)]
                    if safe:
                        self._answer += safe
                        self._render_live()
                        self._buf = self._buf[len(safe):]
                    return
                first, delim, state = min(hits, key=lambda h: h[0])
                if first > 0:                      # render text before the delimiter
                    self._answer += self._buf[:first]
                    self._render_live()
                self._buf   = self._buf[first + len(delim):]
                if state == "drop":
                    continue                      # swallow structural special tokens
                self._finalize_live()             # leaving the answer block → commit its Markdown
                self._state = state
                if state == "thinking":
                    self._think_start = self._time.monotonic()
                    if not self._verbose:
                        # Animated spinner so a collapsed reasoning block doesn't look frozen.
                        self._status = console.status("[dim]Thinking…[/dim]", spinner="dots")
                        self._status.start()
                elif state == "tool":
                    # Tool-call block is suppressed; show a spinner so the call doesn't look frozen.
                    self._status = console.status("[dim]Invoking tool…[/dim]", spinner="dots")
                    self._status.start()
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
                    self._stop_status()
                    console.print(f"[dim]● Thought for {elapsed:.1f}s[/dim]\n", markup=True)
                continue

            elif self._state == "tool":
                # Suppress until the close delimiter; discard the whole span.
                idx = self._buf.find(self._tool_close) if self._tool_close else -1
                if idx != -1:
                    self._stop_status()   # tool block done decoding; drop the spinner
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
    "calculate": "🧮", "prompt_user": "💬",
    "load_tools": "🔧", "load_skill": "📚",
}
_DEFAULT_ICON = "⚙"


# ---------------------------------------------------------------------------
# Shell command classifier — maps idiom commands to (icon, past_label, detail)
# ---------------------------------------------------------------------------

# Each entry: (regex matching the command verb/prefix, icon, past_label, detail_fn)
# detail_fn receives the remainder of the command string after the verb.
_CMD_PATTERNS: list[tuple] = [
    # calculate: PowerShell math / common math funcs / python -c / bc — show the whole expression
    (re.compile(r"^\s*(?=.*(?:\[math\]::|::(?:Sqrt|Pow|Abs|Round|Floor|Ceiling|Log|Exp|Sin|Cos|Tan|Min|Max|PI|E)\b|\bpython3?\s+-c|\bbc\b))", re.I),
     "🧮", "Calculated",
     lambda rest: rest.strip()[:80] or "…"),
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
                elif label == "Calculated":  # show the computed value (drop the "exit 0" prefix)
                    value = result_str.split("\n", 1)[-1].strip()
                    suffix = f" [dim]= {value}[/dim]"
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
        f"\n[bold]Select the folder from which the agent will be working.[/bold]"
    )
    # Try native folder dialog
    path = None
    dialog_shown = False
    try:
        import tkinter as tk
        from tkinter import filedialog
        root_win = tk.Tk(); root_win.withdraw()
        # Force to the foreground; else it opens behind a backgrounded terminal and looks hung.
        root_win.attributes("-topmost", True)
        root_win.update()
        picked = filedialog.askdirectory(initialdir=default,
                                         title="Select project root", parent=root_win)
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


class _RevertSuggest(AutoSuggest):
    """Inline ghost text for the `/revert` command: once the user starts typing it, show the
    numbered list of files that can be reverted (e.g. `vert all  ·  1=cli.py 2=ui.py`). The hint is
    read-only guidance — accepting it is harmless since apply_revert ignores non-numeric tokens.

    get_files() returns the live (orig, bak) backup pairs; an empty list means nothing to suggest.
    """
    def __init__(self, get_files: Callable[[], list]):
        self._get_files = get_files

    def get_suggestion(self, buffer, document):
        text = document.text
        cmd = "/revert"
        # Only fire while typing the bare command (a prefix of "/revert", no args yet). The empty
        # buffer is left to the placeholder; once args are typed the user knows what they want.
        if not text or not cmd.startswith(text.lower()):
            return None
        files = self._get_files() or []
        if not files:
            return None
        hint = "  ".join(f"{k+1}={os.path.basename(orig)}" for k, (orig, _b) in enumerate(files))
        return Suggestion(cmd[len(text):] + f" all  ·  {hint}")


class _GradeSuggest(AutoSuggest):
    """Inline ghost text for `/grade`: active while the previous task's grade window is open.
    get_gradeable() returns True when /grade is accepted, False otherwise."""
    def __init__(self, get_gradeable: Callable[[], bool]):
        self._get_gradeable = get_gradeable

    def get_suggestion(self, buffer, document):
        text = document.text
        cmd = "/grade"
        if not text or not cmd.startswith(text.lower()):
            return None
        if not self._get_gradeable():
            return None
        return Suggestion(cmd[len(text):] + "  ·  grade this task")


class _AnySuggest(AutoSuggest):
    """First non-None suggestion among several — prompt_toolkit accepts a single auto_suggest, but we
    want both /revert and /grade ghost text live at once."""
    def __init__(self, suggesters: list):
        self._suggesters = suggesters

    def get_suggestion(self, buffer, document):
        for s in self._suggesters:
            r = s.get_suggestion(buffer, document)
            if r is not None:
                return r
        return None


async def read_prompt(session: PromptSession, suggest_revert: Callable[[], list] | None = None,
                      placeholder: str | None = None,
                      suggest_grade: Callable[[], float | None] | None = None) -> str | None:
    """Read a user prompt. Returns None only on EOF (Ctrl-D → quit); "" on empty input or Ctrl-C
    so the caller can re-prompt without quitting.

    suggest_revert / suggest_grade / placeholder are passed per-call (not stored on the session) so
    the agent-facing tool prompts that reuse `session` stay clean: placeholder = gray dummy text shown
    while the box is empty; suggest_revert = revertable files for the /revert ghost text;
    suggest_grade = the pending self-grade for the /grade ghost text.

    Async because the REPL runs inside asyncio.run(_repl): prompt_toolkit's sync
    .prompt() would call asyncio.run() again and fail on the running loop. prompt_async
    is the supported in-loop entry point.
    """
    # Tint the typed text so a submitted user turn stays visually distinct in the scrollback.
    kwargs = {"lexer": SimpleLexer(style=_USER_TURN_STYLE)}
    suggesters = ([_RevertSuggest(suggest_revert)] if suggest_revert is not None else []) + \
                 ([_GradeSuggest(suggest_grade)] if suggest_grade is not None else [])
    if suggesters:
        kwargs["auto_suggest"] = _AnySuggest(suggesters)
    if placeholder:
        # Pass as a styled (style, text) tuple, not an HTML f-string: the text is literal, so a
        # placeholder containing XML-reserved chars (&, <, >) won't blow up minidom's parser.
        kwargs["placeholder"] = FormattedText([("fg:ansibrightblack", placeholder)])
    try:
        return (await session.prompt_async(HTML("<ansicyan><b>❯ </b></ansicyan>"), **kwargs)).strip()
    except EOFError:
        return None   # Ctrl-D → quit
    except KeyboardInterrupt:
        return ""     # Ctrl-C at prompt → treat as empty, re-prompt
