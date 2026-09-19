"""wrapper.py — `roger <runtime> [args…]`: the console-script entry point.

Runs a local OpenAI-compatible runtime server exactly as the user typed it, with roger's reverse
proxy (runtime/proxy.py) in front so every chat that flows through it is saved (runtime/capture.py),
the federation's daily global update attached as a LoRA adapter on the runtime's own flag
(runtime/adapter.py), and, after the runtime exits, a training round over the saved chats
(runtime/train.py). `roger train` still reaches the legacy apps/cli.py; nothing else does.

Usage:
  roger llama-server -m model.gguf --port 8080
  roger vllm serve meta-llama/Llama-3-8B --port 8000
  roger train …                               # legacy LoRA update over ~/.roger/runs
"""
import shutil, signal, subprocess, sys, threading

from roger.runtime import adapter, capture, dialect, grader, notice, proxy, train

_USAGE = """usage: roger <runtime-command> [args…]     (e.g. roger llama-server -m x.gguf --port 8080)
       roger train [--batch N] [--epochs N] [--lr F]

Runs the runtime as-is and relays its port; chats are saved under ~/.roger/messages/<runtime>/.
On the first launch of each day the federation's model update is pulled, and every launch attaches it to
the runtime as a LoRA adapter (llama-server --lora / vllm --lora-modules); the model file is never touched.
Once the runtime is up, tells you whether your federations train the model it serves (and which ones they do).
After the runtime exits, once enough graded chats have piled up ("train_every"), trains that adapter on
them locally and contributes the update (Ctrl-C skips); contributed chats are then deleted."""


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

    provider = dialect.runtime_name(binary)
    served = dialect.served_model([binary, *argv[1:]])
    # The federation's update rides along as a LoRA adapter on the runtime's own flag: pulled once a
    # day, rebuilt from disk every launch, the model file untouched. Fail-soft — the runtime starts
    # either way, at worst without today's update.
    try:
        built = adapter.prepare([binary, *argv[1:]])
    except Exception as e:
        built = None
        print(f"roger: federation update not attached ({e.__class__.__name__}: {e})", file=sys.stderr)
    if built is not None:
        p = dialect.attach_adapter(p, binary, *built)
    backend = f"http://127.0.0.1:{p.backend_port}"
    registry: list[dict] = []                          # conversations seen this session (grader.track)
    # Bind before spawning the child so a taken port never leaves an orphaned runtime behind.
    try:
        srv = proxy.start(p.public_host, p.public_port, backend,
                          capture.make_sink(provider, lambda path, rec: grader.track(registry, path, rec),
                                            served),
                          p.request_model)
    except OSError as e:
        print(f"roger: cannot listen on {p.public_host}:{p.public_port} ({e.strerror or e}) — is "
              f"{argv[0]} already running there? Stop it or pass a different --port.", file=sys.stderr)
        return 1
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    stop = threading.Event()
    threading.Thread(target=grader.run, args=(registry, backend, stop, p.request_model), daemon=True).start()
    print(f"roger: relaying {p.public_host}:{p.public_port} → 127.0.0.1:{p.backend_port}; "
          f"saving chats to {capture.messages_dir(provider)}; self-grading chats idle for "
          f"{grader.IDLE_S // 60} min", file=sys.stderr)

    # Inherit stdio so the runtime's output is exactly as if launched directly, but put it in its
    # own process group: the terminal's Ctrl-C must reach only roger, which grades the still-open
    # conversations while the runtime is up and only then forwards the interrupt (_stop).
    child = subprocess.Popen([binary, *p.child_argv[1:]],
                             **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                                if sys.platform == "win32" else {"start_new_session": True}))
    # Once the runtime answers, say whether the federation trains the model it serves (and which models it
    # does accept). Off-thread: the relay must not wait on a model load or a federation round-trip.
    threading.Thread(target=notice.run, args=(backend, lambda: child.poll() is None), daemon=True).start()
    try:
        rc = _wait(child)
    except KeyboardInterrupt:
        stop.set()
        try:
            grader.grade_pending(registry, backend, p.request_model)
        except KeyboardInterrupt:                      # second Ctrl-C: skip the remaining grades
            print("roger: skipping remaining self-grades", file=sys.stderr)
        rc = _stop(child)
    finally:
        proxy.stop(srv)
    # The runtime is gone and its VRAM free: train on what has piled up, if enough has. Never changes
    # the exit status — that stays the runtime's.
    try:
        from roger.apps import config
        train.maybe_train(provider, served, config.load())
    except KeyboardInterrupt:
        print("roger: training skipped; the conversations are kept for next time.", file=sys.stderr)
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
