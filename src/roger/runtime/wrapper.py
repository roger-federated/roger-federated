"""wrapper.py — `roger <runtime> [args…]`: the console-script entry point.

Runs a local OpenAI-compatible runtime server exactly as the user typed it, with roger's reverse
proxy (runtime/proxy.py) in front so every chat that flows through it is saved (runtime/capture.py).
`roger train` still reaches the legacy apps/cli.py; nothing else does.

Usage:
  roger llama-server -m model.gguf --port 8080
  roger vllm serve meta-llama/Llama-3-8B --port 8000
  roger train …                               # legacy LoRA update over ~/.roger/runs
"""
import os, shutil, subprocess, sys, threading

from roger.runtime import capture, dialect, notice, proxy

_USAGE = """usage: roger <runtime-command> [args…]     (e.g. roger llama-server -m x.gguf --port 8080)
       roger train [--batch N] [--epochs N] [--lr F]

Runs the runtime as-is and relays its port; chats are saved under ~/.roger/messages/<runtime>/.
Once the runtime is up, tells you whether your federations train the model it serves (and which ones they do)."""


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
    """After Ctrl-C. The child shares our console/process group so it already got the interrupt;
    give it time to shut down on its own before escalating."""
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
    p = dialect.plan(argv)
    if p is None:
        # Not a server we can sit in front of: be loud that this session contributes nothing, and say
        # what would — the alternative is a user chatting all session with nothing saved.
        print(f"roger: INACTIVE — {argv[0]} has no --port on its command line, so there is nothing to "
              "relay and no chats will be saved. Running it as-is.\n"
              "roger: to contribute, run an OpenAI-compatible server with an explicit --port, e.g.\n"
              "roger:   roger llama-server -m model.gguf --port 8080\n"
              "roger:   roger vllm serve <hf-model> --port 8000\n"
              "roger: (ollama and LM Studio are not supported.)", file=sys.stderr)
        return _passthrough([binary, *argv[1:]])

    provider = os.path.basename(binary).removesuffix(".exe")
    # Bind before spawning the child so a taken port never leaves an orphaned runtime behind.
    try:
        srv = proxy.start(p.public_host, p.public_port, f"http://127.0.0.1:{p.backend_port}",
                          capture.make_sink(provider))
    except OSError as e:
        print(f"roger: cannot listen on {p.public_host}:{p.public_port} ({e.strerror or e}) — is "
              f"{argv[0]} already running there? Stop it or pass a different --port.", file=sys.stderr)
        return 1
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"roger: relaying {p.public_host}:{p.public_port} → 127.0.0.1:{p.backend_port}; "
          f"saving chats to {capture.messages_dir(provider)}", file=sys.stderr)

    # Inherit stdio and the console/process group (no CREATE_NEW_PROCESS_GROUP): the runtime's own
    # output and Ctrl-C handling stay exactly as if it had been launched directly.
    child = subprocess.Popen([binary, *p.child_argv[1:]])
    # Once the runtime answers, say whether the federation trains the model it serves (and which models it
    # does accept). Off-thread: the relay must not wait on a model load or a federation round-trip.
    threading.Thread(target=notice.run, args=(f"http://127.0.0.1:{p.backend_port}", lambda: child.poll() is None),
                     daemon=True).start()
    try:
        rc = _wait(child)
    except KeyboardInterrupt:
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
