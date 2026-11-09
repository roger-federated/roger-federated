"""console.py — roger's own lines on the terminal it shares with the runtime.

The runtime inherits roger's stderr, so roger's handful of lines land between hundreds of the runtime's
own (vllm alone prints a screenful while loading). A coloured `ROGER` badge on every line is what lets
them be found again.
"""
import functools, os, sys

# Bright white on magenta: a badge rather than coloured text, so it stands out from the runtime's own
# coloured level tags (vllm/llama.cpp use green/yellow/red), none of which are magenta.
_BADGE = "\x1b[1;97;45m ROGER \x1b[0m "


def say(msg: str, file=None) -> None:
    """Print `msg` with the ROGER tag on each of its lines. `file` defaults to the stderr of the moment
    (not of import time, so a test's captured stderr is honoured)."""
    out = file or sys.stderr
    tag = _BADGE if _colours(out) else "ROGER: "
    print("\n".join(tag + line for line in msg.split("\n")), file=out)


def _colours(stream) -> bool:
    # Escape codes only for a human at a terminal: piped or redirected output keeps the plain tag so a
    # log stays greppable, and NO_COLOR is the cross-tool opt-out convention (any non-empty value).
    isatty = getattr(stream, "isatty", None)
    if os.environ.get("NO_COLOR") or not (isatty and isatty()):
        return False
    return sys.platform != "win32" or _windows_vt()


@functools.cache
def _windows_vt() -> bool:
    # A Windows console only interprets escape codes once asked to (ENABLE_VIRTUAL_TERMINAL_PROCESSING,
    # 0x4). A legacy console that refuses gets the plain tag instead of raw `←[1;97;45m` garbage.
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-12)   # STD_ERROR_HANDLE: every roger line goes to stderr
        mode = ctypes.c_uint32()
        return bool(k32.GetConsoleMode(h, ctypes.byref(mode)) and k32.SetConsoleMode(h, mode.value | 0x4))
    except Exception:
        return False
