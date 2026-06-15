"""cli.py — Roger Federated CLI entry point.

Usage:
  roger                                    # use config defaults
  roger --model google/gemma-4-E2B-it
  roger --max-steps 20 --verbose
  roger --no-rag --no-skills

Config lives at ~/.roger/config.json; CLI flags override per-run only.
"""
import argparse, asyncio, os, sys, ctypes
from typing import Callable

from rich.console import Console

from roger.apps import config, ui
import roger.serving.model_setup as model_setup
from roger.serving.model_setup import fetch_model
from roger.agency.rollout_utils import rollout

console = Console(highlight=False)

_BANNER = """[bold cyan]
  Roger Federated:[/bold cyan] [dim] local agentic RL[/dim]
  Config: [cyan]{cfg_path}[/cyan]  |  Model: [cyan]{model}[/cyan]
  Change your configuration any time by editing [cyan]{cfg_path}[/cyan].
  Type your task and press Enter. Ctrl-D to quit the session.
  Tip: Plug in your computer before starting a long-running task and disable sleep in your system settings.
  Tip: Follow-up tasks reuse the model's context (KV-cache), so keep related tasks in one session.
"""

# ---------------------------------------------------------------------------
# Keep-awake — prevent OS sleep during a long rollout
# ---------------------------------------------------------------------------

def _keep_awake_start() -> Callable | None:
    """Prevent the OS from sleeping. Returns a cleanup callable (or None)."""
    if sys.platform == "darwin":
        # caffeinate -i runs in background; kill it on cleanup
        import subprocess
        proc = subprocess.Popen(["caffeinate", "-i"])
        return proc.terminate
    elif sys.platform == "win32":
        # SetThreadExecutionState: ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        ES_CONTINUOUS        = 0x80000000
        ES_SYSTEM_REQUIRED   = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        def _restore():
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        return _restore
    else:
        # Linux: try systemd-inhibit; silently skip if unavailable
        try:
            import subprocess
            proc = subprocess.Popen(
                ["systemd-inhibit", "--what=sleep:idle", "--who=roger",
                 "--why=agent rollout", "--mode=block", "sleep", "infinity"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return proc.terminate
        except FileNotFoundError:
            return None


# ---------------------------------------------------------------------------
# Async REPL
# ---------------------------------------------------------------------------

async def _repl(cfg: dict, root: str) -> None:
    """Load model once, then loop: read prompt → rollout → record."""
    # Spinner while loading
    with console.status("[bold]Loading model…[/bold]", spinner="dots"):
        model, processor = fetch_model(cfg["model_id"])
        # Speculative decoding: a user-set draft_model (must share the target's tokenizer) is used
        # as the assistant model; otherwise fall back to model-free n-gram prompt-lookup.
        draft_id = cfg.get("draft_model")
        drafter = model_setup.load_drafter(draft_id, processor.tokenizer) if draft_id else None
        gen_kwargs = ({"assistant_model": drafter} if drafter
                        else {"prompt_lookup_num_tokens": 10})
    tokenizer    = processor.tokenizer
    if not draft_id:
        console.print("[yellow]No draft_model set; using n-gram prompt-lookup. Pass --draft-model "
                      "(saved to config) for faster speculative decoding.[/yellow]")
    elif drafter is None:
        console.print(f"[yellow]draft_model '{draft_id}' has a different tokenizer than the target; "
                      "ignoring it and using n-gram prompt-lookup.[/yellow]")
    # Decode delimiter token-ids back to strings for the text renderer; derive think-channel once
    tool_delims  = tuple(tokenizer.decode([t]) for t in model_setup.find_tool_call_tokens(tokenizer))
    think_delims = model_setup.find_think_delims(tokenizer)
    msg, style = model_setup.placement_summary(model)
    console.print(f"[bold {style}]{msg}[/bold {style}]\n")

    # Prompt toolkit session (history + styled input)
    session    = ui.make_session()
    pt_backend = ui.make_prompt_backend(session)
    renderer   = ui.StreamRenderer(verbose=cfg["verbose"],
                                   think_delims=think_delims, tool_delims=tool_delims)

    async def read_turn(preamble: str = "") -> str | None:
        """Next user turn for the rollout. Surfaces any pending revert notice first; None on Ctrl-D."""
        if preamble:
            console.print()
            console.print(preamble, style="dim", markup=False)  # paths may contain []; don't parse markup
        return await ui.read_prompt(session)

    # First task: read until a non-empty prompt (or quit on Ctrl-D)
    while True:
        text = await read_turn()
        if text is None:         # Ctrl-D
            console.print("\n[dim]Goodbye.[/dim]")
            return
        if text:
            break

    console.print()  # blank line before output
    # A single rollout owns the whole session: it awaits each subsequent user turn itself (via
    # read_turn) and injects it onto the live context, so the KV-cache is reused across tasks rather
    # than rebuilt per message. It returns only when the user quits (Ctrl-D).
    await rollout(
        model, tokenizer, text,
        max_steps      = cfg["max_steps"],
        max_new_tokens = cfg["max_new_tokens"],
        on_token       = renderer.feed,
        on_tool_call   = ui.render_tool_call,
        on_tool_result = ui.render_tool_result,
        prompt_backend = pt_backend,
        read_turn      = read_turn,
        root           = root,
        enable_rag     = cfg["enable_rag"],
        rag_k          = cfg["rag_k"],
        enable_skills  = cfg["enable_skills"],
        enable_memory  = cfg["enable_memory"],
        gen_kwargs     = gen_kwargs,
    )
    console.print("\n[dim]Goodbye.[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="roger",
                                     description="Roger Federated — local agentic RL")
    parser.add_argument("--model",          help="HuggingFace model ID")
    parser.add_argument("--draft-model",    help="HF model ID for the speculative-decoding draft model")
    parser.add_argument("--max-steps",      type=int,  help="Max tool-call turns per run")
    parser.add_argument("--max-new-tokens", type=int,  help="Max generated tokens per turn")
    parser.add_argument("--no-rag",         action="store_true", help="Disable RAG")
    parser.add_argument("--no-skills",      action="store_true", help="Disable skills")
    parser.add_argument("--no-memory",      action="store_true", help="Disable persistent memory")
    parser.add_argument("--verbose",        action="store_true", help="Expand thinking blocks")
    args = parser.parse_args()

    # Load config; apply flag overrides
    cfg = config.load()
    # --draft-model is persisted (unlike the other per-run flags): setting it once carries to
    # future sessions. Saved before the transient overrides so only draft_model is written.
    if args.draft_model:
        cfg["draft_model"] = args.draft_model
        config.save(cfg)
    first_run = not os.path.exists(config.path())  # path existed before load() — inaccurate here;
    # (first-run detection: load() creates the file, so check size afterwards instead)
    if args.model:          cfg["model_id"]      = args.model
    if args.max_steps:      cfg["max_steps"]      = args.max_steps
    if args.max_new_tokens: cfg["max_new_tokens"] = args.max_new_tokens
    if args.no_rag:         cfg["enable_rag"]     = False
    if args.no_skills:      cfg["enable_skills"]  = False
    if args.no_memory:      cfg["enable_memory"]  = False
    if args.verbose:        cfg["verbose"]         = True

    # Banner
    console.print(_BANNER.format(cfg_path=config.path(), model=cfg["model_id"]))

    # Root selection
    root = ui.select_root(os.getcwd())
    os.chdir(root)  # shell tools, file tools, RAG all key off process cwd

    # Keep-awake (released on exit)
    _wake_cleanup = _keep_awake_start()

    try:
        asyncio.run(_repl(cfg, root))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    finally:
        if _wake_cleanup:
            _wake_cleanup()


if __name__ == "__main__":
    main()
