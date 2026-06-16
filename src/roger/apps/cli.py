"""cli.py — Roger Federated CLI entry point.

Usage:
  roger                                    # use config defaults
  roger --model google/gemma-4-E2B-it
  roger --max-steps 20 --verbose
  roger --no-rag --no-skills

Config lives at ~/.roger/config.json; CLI flags override per-run only.
"""
import argparse, asyncio, os, sys, ctypes, warnings
from typing import Callable

from rich.console import Console

from roger.apps import config, ui
import roger.serving.model_setup as model_setup
from roger.serving.model_setup import fetch_model
from roger.agency.rollout_utils import rollout
from roger.tools import mcp_utils

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
# Console output policy — clean by default, raw under --verbose
# ---------------------------------------------------------------------------

def _configure_output(verbose: bool) -> None:
    """Quiet third-party noise unless --verbose. Our own warnings are always shown, pretty-printed."""
    warnings.simplefilter("always")  # no per-location dedup — recurring notices fire every time
    if verbose:
        return  # leave defaults: raw warnings + tqdm bars visible for debugging
    # Match the real package dir, not the substring "roger" (the uv tool dir is itself named
    # roger, so third-party packages live under …/roger/.../site-packages/torch/… too).
    import roger
    pkg = os.path.dirname(os.path.abspath(roger.__file__))
    def _show(message, category, filename, lineno, file=None, line=None):
        if filename and os.path.abspath(filename).startswith(pkg):
            console.print(f"[yellow]⚠ {category.__name__}: {message}[/yellow]")
    warnings.showwarning = _show
    # Silence hub download bars, the transformers weight-loading bar + advisory logging, and
    # torch's "triton not found" log. These take effect for the later fetch.
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    import transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    import logging
    logging.getLogger("torch").setLevel(logging.ERROR)


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
        sliding = model_setup.uses_sliding_window(model)
        # n-gram crashes on sliding-window models because it crops KV-cache
        gen_kwargs = ({"assistant_model": drafter} if drafter # TODO: may still crash due to sliding window
                        else {} if sliding
                        else {"prompt_lookup_num_tokens": 10})
    tokenizer = processor.tokenizer
    if drafter:
        pass  # user's draft model in use
    elif draft_id:
        console.print(f"[yellow]draft_model '{draft_id}' has a different tokenizer than the target; "
                      "ignoring it.[/yellow]")
    elif sliding:
        console.print("[yellow]n-gram speculative decoding disabled: model uses sliding-window "
                      "attention (its KV-cache can't be cropped). Pass --draft-model for a drafter.[/yellow]")
    else:
        console.print("[yellow]No draft_model set; using n-gram prompt-lookup. Pass --draft-model "
                      "(auto-saved to config) for faster speculative decoding.[/yellow]")
    # Decode delimiter token-ids back to strings for the text renderer; derive think-channel once
    tool_delims  = tuple(tokenizer.decode([t]) for t in model_setup.find_tool_call_tokens(tokenizer))
    think_delims = model_setup.find_think_delims(tokenizer)
    # All special-token strings: the renderer drops any that surface in answer text (turn close,
    # tool_response, eos, …) — the model emits them structurally but they must not be printed.
    specials = [tokenizer.decode([t]) for t in tokenizer.all_special_ids]
    msg, style = model_setup.placement_summary(model)
    console.print(f"[bold {style}]{msg}[/bold {style}]\n")

    # Prompt toolkit session (history + styled input)
    session    = ui.make_session()
    pt_backend = ui.make_prompt_backend(session)
    renderer   = ui.StreamRenderer(verbose=cfg["verbose"], think_delims=think_delims,
                                   tool_delims=tool_delims, specials=specials)

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

    # Connect any MCP servers declared in ~/.roger/mcp.json (stdio or remote). 
    # The stack stays open for the whole session and is closed in the finally below. 
    mcp_servers = mcp_utils.load_mcp_config()
    mcp_stack, mcp_tools, mcp_handlers = None, [], {}
    if mcp_servers:
        with console.status("[bold]Connecting MCP servers…[/bold]", spinner="dots"):
            mcp_stack, mcp_tools, mcp_handlers = await mcp_utils.connect_servers(mcp_servers)
        console.print(f"[dim]MCP: {len(mcp_handlers)} tool(s) from "
                      f"{len(mcp_servers)} server(s).[/dim]")

    console.print()  # blank line
    # A single rollout owns the whole session: it awaits each subsequent user turn itself (via
    # read_turn) and injects it onto the live context, so the KV-cache is reused across tasks rather
    # than rebuilt per message. It returns only when the user quits (Ctrl-D).
    try:
        await rollout(
            model, tokenizer, text,
            tools          = mcp_tools,
            tool_handlers  = mcp_handlers,
            max_steps      = cfg["max_steps"],
            max_new_tokens = cfg["max_new_tokens"],
            on_token       = renderer.feed,
            on_gen_start   = renderer.reset,   # per-turn renderer lifecycle: reset at start, flush tail at end
            on_gen_end     = renderer.flush,
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
    finally:
        if mcp_stack is not None:
            await mcp_stack.aclose()
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

    # Console output policy (before any model fetch so download bars are suppressed)
    _configure_output(cfg["verbose"])

    # Wipe any leftover terminal content before the session (after output config so it isn't garbled)
    console.clear()

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
