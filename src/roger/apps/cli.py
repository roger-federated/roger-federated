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
import roger.loading.model_setup as model_setup
from roger.loading.model_setup import fetch_model
from roger.agency.path_utils import state_dir
from roger.agency.rollout_utils import rollout
from roger.tools import mcp_utils, std_tools

console = Console(highlight=False)

_BANNER = """
[bold cyan]Roger Federated:[/bold cyan] [dim] local agentic RL[/dim]
Config: [cyan]{cfg_path}[/cyan]  |  Model: [cyan]{model}[/cyan]
Tip 1: Before starting a long-running task, plug in your computer and disable sleep in your system settings.
Tip 2: To avoid polluting the KV-cache, provide instructions incrementally and start a fresh session for new tasks.
"""

# Shown when a federation reports this client is out of date. There is no PyPI release (readme.md):
# roger is installed from a local clone, so the update is a git pull + reinstall of the tool. We can't
# know where the user cloned it, so the instruction names the repo rather than a path.
_UPDATE_CMD = "git pull && uv tool install . --reinstall   (in your roger-federated clone)"

_PRIVACY_URL = "https://github.com/roger-federated/roger-federated/blob/main/PRIVACY.md"


def _ensure_privacy_notice(cfg: dict) -> None:
    """One-time notice, before this client's first-ever federation activity, of what gets sent to the
    default federation server and how to opt out. Sentinel in state_dir() so it shows once per machine,
    not once per run. Not gated on a keypress: the processing basis is legitimate interest with a
    standing right to object (opt out), not consent, so transparency is what's required, not sign-off."""
    if not cfg.get("federations"):
        return
    sentinel = os.path.join(state_dir(), "privacy_ack")
    if os.path.exists(sentinel):
        return
    console.print(
        "[yellow]Roger contributes an encrypted, secret-shared gradient update to your configured "
        "federation server(s) by default (never raw data). The server also logs ordinary connection "
        f"metadata (IP, timestamp) for every request, like any web service. Details: {_PRIVACY_URL}\n"
        "Opt out anytime by setting \"federations\": [] in ~/.roger/config.json.[/yellow]\n")
    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    open(sentinel, "w").close()

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
    # The federation's cumulative global ΔW (if any) is folded into the base weights in bf16 before
    # quantization — see federated/delta.fold_into + loading/model_setup. The HF cache is untouched;
    # nothing is stored. None when no federation is configured or none has been pulled yet.
    from roger.federated import client as fed_client
    global_deltas = fed_client.pending_globals(cfg)
    # Spinner while loading
    with console.status("[bold]Loading model…[/bold]", spinner="dots"):
        model, processor = fetch_model(cfg["model_id"], weight_deltas=global_deltas)
        # Speculative decoding: a user-set draft_model (must share the target's tokenizer) is used
        # as the assistant model; otherwise fall back to model-free n-gram prompt-lookup.
        # Off by default on sliding-window models ("auto") as it adds output_hidden_states overhead;
        # set "speculative": true to opt in.
        draft_id = cfg.get("draft_model")
        sliding  = model_setup.uses_sliding_window(model)
        spec     = cfg.get("speculative", "auto")
        speculate = spec if isinstance(spec, bool) else not sliding   # auto: off on sliding
        drafter   = (model_setup.load_drafter(draft_id, processor.tokenizer) if draft_id else None) if speculate else None
        gen_kwargs = ({"assistant_model": drafter} if drafter
                        else {"prompt_lookup_num_tokens": 10} if speculate
                        else {})
    tokenizer = processor.tokenizer
    if not speculate:
        if sliding:
            console.print("[dim]Speculative decoding off (sliding-window model). "
                          "Set \"speculative\": true in the config to enable it.[/dim]")
    elif drafter:
        if sliding:
            console.print("[dim]Speculative decoding: drafter active.[/dim]")
        # else: non-sliding drafter; nothing to flag
    elif draft_id:
        console.print(f"[yellow]draft_model '{draft_id}' has a different tokenizer than the target; "
                      "falling back to n-gram prompt-lookup.[/yellow]")
    else:
        console.print("[yellow]No draft_model set; using n-gram prompt-lookup. Pass --draft-model "
                      "(or set \"draft_model\" in the config) for faster speculative decoding.[/yellow]")
    # Decode delimiter token-ids back to strings for the text renderer; derive think-channel once
    tool_delims  = tuple(tokenizer.decode([t]) for t in model_setup.find_tool_call_tokens(tokenizer))
    _think_ids   = model_setup.find_think_tokens(tokenizer)
    think_delims = tuple(tokenizer.decode([t]) for t in _think_ids) if _think_ids else None
    # All special-token strings: the renderer drops any that surface in answer text (turn close,
    # tool_response, eos, …) — the model emits them structurally but they must not be printed.
    specials = [tokenizer.decode([t]) for t in tokenizer.all_special_ids]
    msg, style = model_setup.placement_summary(model)
    console.print(f"[bold {style}]{msg}[/bold {style}]\n")
    if global_deltas:
        console.print(f"[dim]Folded federated update into {len(global_deltas)} layers.[/dim]")

    # Prompt toolkit session (history + styled input)
    session    = ui.make_session()
    pt_backend = ui.make_prompt_backend(session)
    renderer   = ui.StreamRenderer(verbose=cfg["verbose"], think_delims=think_delims,
                                   tool_delims=tool_delims, specials=specials)

    # Gray dummy task shown as ghost text while the input box is empty (startup + idle turns).
    # The Ctrl-D hint matters: quitting that way is what triggers the model's memory-save turn
    placeholder = ("Ask Roger to do something…  (Ctrl-D to quit and save memory)"
                   if cfg["enable_memory"] else "Ask Roger to do something…  (Ctrl-D to quit)")

    async def read_turn(preamble: str = "") -> str | None:
        """Next user turn for the rollout. Surfaces any pending revert notice first; None on Ctrl-D."""
        if preamble:
            console.print()
            console.print(preamble, style="dim", markup=False)  # paths may contain []; don't parse markup
        return await ui.read_prompt(session, suggest_revert=std_tools.pending_backups,
                                    placeholder=placeholder, suggest_grade=std_tools.gradeable)

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
            model, processor, text,
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

    # Gated quit-time training: inference is over, so reuse the loaded model when runs have piled up.
    # Only trains when contributing into a federation (there is no local-apply path); the resulting
    # ΔW is densified, masked and uploaded, never written to ~/.roger/adapter.
    from roger.training import trainer
    from roger.federated import client as fed_client
    train_every = cfg.get("train_every", 8)
    if fed_client.should_train(cfg) and len(trainer._list_unconsumed(train_every)) >= train_every:
        feds = cfg.get("federations") or []
        # Don't burn the quit-time update on a gradient no federation will accept: if every configured
        # federation would reject it — allowlist excludes this model, or requires a newer client than we
        # speak — skip training but KEEP the runs (discard only ever happens on an accepted upload), so
        # they're there once the user switches model / federation or updates the client.
        statuses = fed_client.probe_federations(cfg)
        blocked = set(fed_client.unsupported_urls(statuses)) | set(fed_client.outdated_urls(statuses))
        if len(blocked) == len(feds):
            console.print(f"[yellow]Skipping training: no configured federation will accept a gradient "
                          f"for {cfg['model_id']} from this client (unsupported model, or the client is "
                          "out of date). Your recorded runs are kept for next time.[/yellow]")
        else:
            with console.status("[bold]Training on recent runs…[/bold]", spinner="dots"):
                stats = trainer.train(model_id=cfg["model_id"], reuse=(model, processor))
            console.print(f"[dim]Training: {stats}[/dim]")
            if stats.get("delta"):
                with console.status("[bold]Contributing your gradient…[/bold]", spinner="dots"):
                    accepted = fed_client.contribute_delta(stats["delta"], cfg)
                if accepted:                    # consume the runs only once the ΔW is actually shared
                    trainer.discard_runs(stats.get("consumed_dirs", []))

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
    parser.add_argument("--no-contribute",  action="store_true",
                        help="Leech mode: pull the federated model but share no gradients this run")
    # Federation servers are managed only via the "federations" list in ~/.roger/config.json
    # (no flag): the source of truth is the config, so a per-run flag can't silently append to or
    # shadow the default server.
    # Optional `train` subcommand; bare `roger` (cmd=None) still launches the chat REPL.
    sub = parser.add_subparsers(dest="cmd")
    p_train = sub.add_parser("train", help="Run a LoRA REINFORCE++ update over recorded runs")
    p_train.add_argument("--batch",  type=int,   default=8, help="Max runs (episodes) per update")
    p_train.add_argument("--epochs", type=int,   default=1, help="Passes over the batch")
    p_train.add_argument("--lr",     type=float, default=1e-5, help="LoRA learning rate")
    args = parser.parse_args()

    # Load config; apply flag overrides. Every flag is a per-run override only — none are written
    # back to ~/.roger/config.json. To change a setting permanently, edit the config file.
    cfg = config.load()
    if args.model:          cfg["model_id"]      = args.model
    if args.draft_model:    cfg["draft_model"]    = args.draft_model
    if args.max_steps:      cfg["max_steps"]      = args.max_steps
    if args.max_new_tokens: cfg["max_new_tokens"] = args.max_new_tokens
    if args.no_rag:         cfg["enable_rag"]     = False
    if args.no_skills:      cfg["enable_skills"]  = False
    if args.no_memory:      cfg["enable_memory"]  = False
    if args.verbose:        cfg["verbose"]         = True
    if args.no_contribute:  cfg["contribute"]      = False

    # Console output policy (before any model fetch so download bars are suppressed)
    _configure_output(cfg["verbose"])

    # `roger train`: standalone update over recorded runs, loading its own training model. The ΔW is
    # shared with the federation, never applied locally — so without a federation there's nothing to do.
    if args.cmd == "train":
        from roger.training import trainer
        from roger.federated import client as fed_client
        _ensure_privacy_notice(cfg)
        if not fed_client.should_train(cfg):
            why = ("you're in leech mode (\"contribute\": false)" if cfg.get("federations")
                   else "no federations are configured in ~/.roger/config.json")
            console.print(f"[yellow]Nothing to train: {why}. Training only produces a gradient to "
                          "contribute; it is never applied locally.[/yellow]")
            return
        with console.status("[bold]Training on recorded runs…[/bold]", spinner="dots"):
            stats = trainer.train(model_id=cfg["model_id"], batch=args.batch, epochs=args.epochs,
                                  lr=args.lr)
        console.print(f"Training: {stats}")
        if stats.get("delta"):
            with console.status("[bold]Contributing your gradient to the federation…[/bold]", spinner="dots"):
                accepted = fed_client.contribute_delta(stats["delta"], cfg)
            if accepted:                        # consume the runs only once the ΔW is actually shared
                trainer.discard_runs(stats.get("consumed_dirs", []))
        return

    # Wipe any leftover terminal content before the session (after output config so it isn't garbled)
    console.clear()

    # Banner
    console.print(_BANNER.format(cfg_path=config.path(), model=cfg["model_id"]))

    # Federated: nudge a leech, then download + persist the day's cumulative global once on first
    # startup. It's folded into the base at model-load time (in _repl). Inert without a federation.
    from roger.federated import client as fed_client
    _ensure_privacy_notice(cfg)
    # Warn if the user has dropped the default federation server: without it they pull no community
    # updates and run the bare base model, so model quality is meaningfully worse.
    _defaults = config.default_federations()
    if _defaults and not any(f in cfg.get("federations", []) for f in _defaults):
        console.print("[yellow]⚠ You're not partaking in the default federation server, so you "
                      "won't pull its updates and will experience degraded model "
                      f"performance. Add it back to \"federations\" in {config.path()}: "
                      f"{_defaults[0]}[/yellow]")
    if fed_client.is_leeching(cfg):
        console.print("[yellow]🪱 Leech mode: you're pulling the federation's model but contributing "
                      "nothing back. Set \"contribute\": true in ~/.roger/config.json to pull your "
                      "weight.[/yellow]")
    if cfg.get("federations"):
        with console.status("[bold]Syncing federated model…[/bold]", spinner="dots"):
            fetched = fed_client.maybe_daily_pull(cfg)
            statuses = fed_client.probe_federations(cfg)   # one /status pass, reused for every verdict
        feds = cfg["federations"]
        unsupported = fed_client.unsupported_urls(statuses)
        outdated = fed_client.outdated_urls(statuses)
        if fetched:
            console.print("[dim]Downloaded today's federation update.[/dim]")
        # Warn up front (before a session's worth of gradient is wasted) if a federation's allowlist
        # excludes the selected model: those feds won't take this session's gradient or serve updates.
        if unsupported:
            scope = ("None of your federations accept" if len(unsupported) == len(feds)
                     else f"{len(unsupported)} of your federations don't accept")
            console.print(f"[yellow]⚠ {scope} {cfg['model_id']}: {', '.join(unsupported)}. "
                          "Pick a supported model or federation to share gradients and receive its "
                          "updates.[/yellow]")
        # Hard incompatibility: a federation requires a newer protocol than this build speaks, so it
        # would reject our gradient. Block contributing to those feds (runs are kept, same as an
        # unsupported model) and tell the user how to update. A merely-newer latest_client is advisory.
        if outdated:
            scope = ("None of your federations accept" if len(outdated) == len(feds)
                     else f"{len(outdated)} of your federations no longer accept")
            console.print(f"[yellow]⚠ Your roger client is out of date: {scope} this version's "
                          f"gradients. Update to keep contributing:\n  {_UPDATE_CMD}[/yellow]")
        elif fed_client.newest_client(statuses):
            console.print(f"[dim]A newer roger client is available. Update with: {_UPDATE_CMD}[/dim]")

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
