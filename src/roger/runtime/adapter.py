"""adapter.py — bring the federation's global update to the model the runtime serves.

The model is loaded by llama-server / vllm, not by roger, so the update can only reach it as an
*adapter* — roger never touches weights. At launch, before the runtime is spawned:
once per UTC day each federation's cumulative global is pulled for the model the command line names and
persisted (federated/transport), and at *every* launch the persisted global is materialised as a LoRA
adapter in the runtime's own format — a GGUF adapter for llama-server, a PEFT directory for vllm —
which runtime/dialect.py writes and attaches with the runtime's flag. The model file / HF cache is never
touched and no model copy is ever stored: the adapter *is* the update, and the runtime applies it at load.

The global is expected in LoRA-factor form (the contract in federated/delta.py: `<module>.lora_A.weight`
[r, in], `<module>.lora_B.weight` [out, r], metadata `scaling`). A blob without factor keys — a global
persisted by an older client, from before the server switched — can't be attached as an adapter, so it
is reported rather than applied.

Torch-free on purpose: the wrapper must start in well under a second (numpy + safetensors + gguf only).
Everything fails soft to "no adapter this launch" with one stderr line — a federation hiccup must never
keep the user's runtime from starting.
"""
import hashlib, json, os, struct, sys
from datetime import datetime, timezone

import numpy as np
from safetensors.numpy import load as st_load

from roger.paths import state_dir
from roger.federated import transport
from roger.runtime import dialect, notice


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def pull_today(feds: list[str], hint: str, out=sys.stderr) -> bool:
    """First launch of the UTC day, per federation: resolve `hint` (what the command line calls the model)
    to the id the federation trains it as, fetch the global if it moved, persist blob + cursor. Returns
    whether anything new arrived. The day is stamped even when nothing came back (unreachable,
    unsupported) so later launches don't re-poll all day: a transient outage costs one day, not 30s of
    connect timeout on every launch while offline."""
    today, fetched = _today(), False
    for url in feds:
        st = transport.load_state(url, hint)
        if st.get("last_sync") == today:
            continue
        status = transport.federation_status(url, hint)
        accepted = status.get("models")           # None: no allowlist (or a server predating the field)
        model_id = notice.resolve([hint], accepted) if accepted is not None else hint
        if status and model_id is not None and status.get("mode", "busy") != "unsupported":
            print(f"roger: pulling today's federation update for {model_id} from {url}…", file=out)
            res = transport.pull(url, st.get("cursor"), model_id)
            if res is not None:
                blob, cursor = res
                transport.save_global(url, blob, hint)
                st["cursor"], fetched = cursor, True
            st["model_id"] = model_id               # the id the adapter is built for (PEFT config)
        st["last_sync"] = today
        transport.save_state(url, st, hint)
    return fetched


def _metadata(buf: bytes) -> dict:
    # safetensors: u64 LE header length, then the JSON header whose "__metadata__" holds our str→str
    # fields. Mirrors federated/delta._read_metadata, which sits behind a torch import.
    n = struct.unpack("<Q", buf[:8])[0]
    return json.loads(buf[8 : 8 + n]).get("__metadata__", {})


def load_factors(feds: list[str], hint: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{module: (A [r, in], B [out, r])} with the update = B@A, from every federation's persisted global
    for `hint`. Federations are joined along the rank axis (B@A of the concatenation = the sum of the
    parts), so one adapter carries them all at scale 1, with each federation's `scaling` and a 1/N merge
    coefficient folded into its B. The 1/N: the globals are cumulated independently and overlap in what
    they learned, so summing N of them at full scale would overshoot; N counts only the federations that
    actually have a factor-form global here, so one with nothing yet doesn't dilute the rest. (Training
    never uses this merge: it attaches only the one federation it contributes to, at scale 1 — see
    runtime/train.py.) A dense (pre-factor) global has no factor keys and contributes nothing."""
    globals_ = []
    for url in feds:
        blob = transport.load_global(url, hint)
        if blob is None:
            continue
        tensors = st_load(blob)
        if any(k.endswith(".lora_A.weight") for k in tensors):
            globals_.append((tensors, float(_metadata(blob).get("scaling", 1.0))))
    out: dict = {}
    for tensors, scaling in globals_:
        coef = scaling / len(globals_)
        for key, A in tensors.items():
            if not key.endswith(".lora_A.weight"):
                continue
            mod = key[: -len(".lora_A.weight")]
            B = tensors.get(mod + ".lora_B.weight")
            if B is None:
                continue
            A, B = A.astype(np.float32), B.astype(np.float32) * coef
            if mod in out:
                A, B = np.concatenate([out[mod][0], A]), np.concatenate([out[mod][1], B], axis=1)
            out[mod] = (A, B)
    return out


def _adapter_stem(hint: str) -> str:
    d = os.path.join(state_dir(), "federated", "adapters")
    os.makedirs(d, exist_ok=True)
    # Keyed by the served model: two runtimes on different models never clobber each other's adapter.
    return os.path.join(d, hashlib.sha1(hint.encode()).hexdigest()[:16])


def prepare(argv: list[str], out=sys.stderr) -> tuple[str, int] | None:
    """The wrapper's one call before spawning the runtime: today's pull (first launch of the day) and the
    adapter to attach for this command line — (path, rank), or None when there's nothing to attach. The
    three ways a command line rules the update out on its own — an unknown runtime, the user's own LoRA,
    and no model named — are warned about rather than passed over silently; none of them is visible from
    the session otherwise. Opting out of the federation is the user's own doing, so it says nothing."""
    from roger import config              # first run writes the default config = default federation
    cfg = config.load()
    feds = cfg.get("federations") or []
    if not feds:
        return None                       # opted out of the federation entirely: nothing to attach
    runtime = dialect.runtime_name(argv[0])
    spec = dialect.RUNTIMES.get(runtime)
    # Each check below costs the user the federation silently if left unsaid: the relay, the capture and
    # the self-grading all work regardless, so the session looks healthy and the loss only shows up as
    # chats that never train. notice.announce makes that worse — it reads the model off /v1/models, so it
    # will report the federation as training this very model while nothing here can train it.
    if spec is None:
        print(f"roger: {runtime} isn't a runtime roger knows ({', '.join(dialect.RUNTIMES)}), so no "
              "federation update can be attached and this session's chats can't be trained on. They're "
              "still saved and graded.", file=out)
        return None
    if (flag := dialect.user_adapter(argv)) is not None:
        print(f"roger: cannot attach the federation's LoRA adapter because {flag} already loads one of "
              "yours; drop it to receive the federation's update. Chats are still saved and graded.",
              file=out)
        return None
    hint = dialect.served_model(argv)
    if hint is None:
        print(f"roger: this {runtime} command line doesn't name a model ({dialect.names_model(spec)}), "
              "so roger can't tell which model is served: no federation update will be attached and "
              "this session's chats can't be trained on. They're still saved and graded.", file=out)
        return None
    notice.privacy_notice(cfg, out)               # before the pull below: the first federation contact
    pull_today(feds, hint, out)
    factors = load_factors(feds, hint)
    if not factors:
        if any(transport.load_global(url, hint) is not None for url in feds):
            print("roger: the federation's global isn't in adapter (LoRA-factor) form yet, so it can't "
                  "be attached to the runtime; chats are still saved.", file=out)
        return None
    model_id = next((m for url in feds if (m := transport.load_state(url, hint).get("model_id"))), hint)
    built = spec["build"](factors, hint, _adapter_stem(hint), model_id, out)
    if built is not None:
        print(f"roger: attaching the federation's update ({len(factors)} modules, rank {built[1]}) "
              f"to {os.path.basename(hint)} via adapter {built[0]}", file=out)
    return built
