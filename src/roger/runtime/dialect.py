"""dialect.py — everything roger knows about specific runtimes, in one place.

The rest of runtime/ is framework-agnostic: proxy.py relays bytes, capture.py reads the OpenAI wire format
every supported server speaks. What differs per runtime lives here — for now the command line: the
`--port`/`--host` convention `plan()` rewrites to slot the proxy in front.
Runtimes with their own conventions (ollama's env-configured port and native NDJSON dialect, LM Studio's
detached `lms server start`) are deliberately not special-cased.
"""
import socket
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Command line: where the runtime goes, where we listen
# ---------------------------------------------------------------------------

class Plan(NamedTuple):
    child_argv: list[str]   # the runtime's argv, `--port` rewritten to the backend port
    public_host: str        # where roger listens (the address clients already use)
    public_port: int
    backend_port: int       # loopback port the real runtime is moved to


def free_port() -> int:
    """Ephemeral loopback port from the OS — self-derived, so it can't collide with the user's choice."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _flag_value(argv: list[str], name: str) -> str | None:
    """Value of `--name X` or `--name=X`; last occurrence wins, like argparse."""
    val = None
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith(name + "="):
            val = a[len(name) + 1:]
    return val


def _set_flag(argv: list[str], name: str, value: str) -> list[str]:
    """Copy of argv with every `--name X` / `--name=X` set to `value` (appended if absent)."""
    out, i, seen = [], 0, False
    while i < len(argv):
        a = argv[i]
        if a == name and i + 1 < len(argv):
            out += [name, value]; i += 2; seen = True
        elif a.startswith(name + "="):
            out.append(f"{name}={value}"); i += 1; seen = True
        else:
            out.append(a); i += 1
    return out if seen else out + [name, value]


def plan(argv: list[str], backend_port: int | None = None) -> Plan | None:
    """Split `<runtime> … --port N [--host H]` into a public listen address and a rewritten child
    command. None when there's no `--port`: nothing tells us where messages would surface, so the
    command is not a server we can wrap (or not a server at all) and should just run as-is."""
    port = _flag_value(argv, "--port")
    if port is None or not port.isdigit():
        return None
    backend = backend_port or free_port()
    child = _set_flag(argv, "--port", str(backend))
    # Only pin the child to loopback when the user *gave* a --host: we can't know whether an unknown
    # runtime even accepts the flag, and every server we care about defaults to loopback anyway.
    # When the user asked for 0.0.0.0 the LAN should reach roger, never the raw runtime behind it.
    host = _flag_value(argv, "--host")
    if host is not None:
        child = _set_flag(child, "--host", "127.0.0.1")
    return Plan(child, host or "127.0.0.1", int(port), backend)
