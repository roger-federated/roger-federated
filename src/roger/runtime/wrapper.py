"""wrapper.py — `roger <runtime> [args…]`: the console-script entry point.

Runs a local OpenAI-compatible runtime server exactly as the user typed it, with roger's reverse
proxy (runtime/proxy.py) in front so every chat that flows through it is saved (runtime/capture.py).
`roger train` still reaches the legacy apps/cli.py; nothing else does.

Usage:
  roger llama-server -m model.gguf --port 8080
  roger vllm serve meta-llama/Llama-3-8B --port 8000
  roger train …                               # legacy LoRA update over ~/.roger/runs
"""
import os, shutil, signal, subprocess, sys, threading

from roger.runtime import capture, grader, proxy

_USAGE = """usage: roger <runtime-command> [args…]     (e.g. roger llama-server -m x.gguf --port 8080)
       roger train [--batch N] [--epochs N] [--lr F]

Runs the runtime as-is and relays its port; chats are saved under ~/.roger/messages/<runtime>/."""


def _passthrough(argv: list[str]) -> int:
    """No server to wrap: run the command untouched. Ctrl-C goes to the child (shared console /
    process group); we just wait for it to act on it rather than dying first."""
    child = subprocess.Popen(argv)
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue


def _wait(child: subprocess.Popen) -> int:
    # Poll rather than a bare wait(): on Windows an infinite WaitForSingleObject swallows Ctrl-C.
    while True:
        try:
            return child.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass


def _stop(child: subprocess.Popen) -> int:
    """After Ctrl-C. The runtime sits in its own process group (so the terminal's interrupt reached
    only us and the pending self-grades could run first); now deliver the interrupt ourselves and
    give it time to shut down on its own before escalating."""
    child.send_signal(signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT)
    try:
        rc = child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        rc = 130
    except KeyboardInterrupt:                          # second Ctrl-C: the user means it
        child.kill()
        child.wait()
        rc = 130
    return 130 if rc < 0 else rc                       # POSIX reports death-by-signal as -SIG


def run(argv: list[str]) -> int:
    binary = shutil.which(argv[0])
    if binary is None:
        print(f"roger: {argv[0]!r} not found on PATH", file=sys.stderr)
        return 127
    p = proxy.plan(argv)
    if p is None:
        print(f"roger: no --port on the command line, so there is no port to relay; "
              f"running {argv[0]} as-is without capture", file=sys.stderr)
        return _passthrough([binary, *argv[1:]])

    provider = os.path.basename(binary).removesuffix(".exe")
    backend = f"http://127.0.0.1:{p.backend_port}"
    registry: list[dict] = []                          # conversations seen this session (grader.track)
    # Bind before spawning the child so a taken port never leaves an orphaned runtime behind.
    try:
        srv = proxy.start(p.public_host, p.public_port, backend,
                          capture.make_sink(provider, lambda path, rec: grader.track(registry, path, rec)))
    except OSError as e:
        print(f"roger: cannot listen on {p.public_host}:{p.public_port} ({e.strerror or e}) — is "
              f"{argv[0]} already running there? Stop it or pass a different --port.", file=sys.stderr)
        return 1
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    stop = threading.Event()
    threading.Thread(target=grader.run, args=(registry, backend, stop), daemon=True).start()
    print(f"roger: relaying {p.public_host}:{p.public_port} → 127.0.0.1:{p.backend_port}; "
          f"saving chats to {capture.messages_dir(provider)}; self-grading chats idle for "
          f"{grader.IDLE_S // 60} min", file=sys.stderr)

    # Inherit stdio so the runtime's output is exactly as if launched directly, but put it in its
    # own process group: the terminal's Ctrl-C must reach only roger, which grades the still-open
    # conversations while the runtime is up and only then forwards the interrupt (_stop).
    child = subprocess.Popen([binary, *p.child_argv[1:]],
                             **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                                if sys.platform == "win32" else {"start_new_session": True}))
    try:
        rc = _wait(child)
    except KeyboardInterrupt:
        stop.set()
        try:
            grader.grade_pending(registry, backend)
        except KeyboardInterrupt:                      # second Ctrl-C: skip the remaining grades
            print("roger: skipping remaining self-grades", file=sys.stderr)
        rc = _stop(child)
    finally:
        proxy.stop(srv)
    return rc


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "train":                    # legacy path; the only reason cli.py is still imported
        from roger.apps import cli
        cli.main()
        return
    if not argv or argv[0].startswith("-"):
        print(_USAGE, file=sys.stdout if argv and argv[0] in ("-h", "--help") else sys.stderr)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
