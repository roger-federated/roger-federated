"""dialect.py — everything roger knows about specific runtimes and their file formats, in one place.

The rest of runtime/ is framework-agnostic: proxy.py relays bytes, capture.py reads the OpenAI wire format
every supported server speaks, adapter.py pulls the federation's update. What differs per runtime lives here:
  - the command line: the `--port`/`--host` convention `plan()` rewrites to slot the proxy in front, and
    where each runtime names its model (`served_model`);
  - how it loads a LoRA: llama.cpp's GGUF adapter (`build_gguf`) and PEFT's directory layout, which vllm
    reads (`build_peft`), plus the arguments that attach one (`attach_adapter`), the flags that say the
    user is already loading one of their own (`user_adapter`) and, for vllm, the model name chat requests
    must be pointed at.
Supporting another runtime means one more `RUNTIMES` entry (and a writer, if its adapter format is new).
Runtimes with their own conventions (ollama's env-configured port and native NDJSON dialect, LM Studio's
detached `lms server start`) are deliberately not special-cased.
"""
import json, os, socket, sys
from typing import NamedTuple

import numpy as np
from safetensors.numpy import save_file as st_save_file

_PEFT_PREFIX = "base_model.model."


# ---------------------------------------------------------------------------
# Command line: where the runtime goes, where we listen
# ---------------------------------------------------------------------------

class Plan(NamedTuple):
    child_argv: list[str]   # the runtime's argv, `--port` rewritten to the backend port
    public_host: str        # where roger listens (the address clients already use)
    public_port: int
    backend_port: int       # loopback port the real runtime is moved to
    request_model: str | None = None   # rewrite chat requests' `model` to this (vllm adapter name)


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


# ---------------------------------------------------------------------------
# LoRA adapter formats
# ---------------------------------------------------------------------------

def _hf_path(module: str) -> str:
    # PEFT keys a module `base_model.model.<hf path>`; the adapter formats speak the HF path.
    return module[len(_PEFT_PREFIX):] if module.startswith(_PEFT_PREFIX) else module


def build_peft(factors: dict, model: str, dest: str, model_id: str, out=sys.stderr) -> tuple[str, int]:
    """A PEFT adapter directory at `dest` (what vllm's `--lora-modules` loads): adapter_config.json + the
    factors under the keys PEFT itself saves. r/alpha are set so PEFT's α/r scale is 1 — the scale already
    lives in B — and r is the largest rank present (vllm only needs an upper bound). `model_id` is recorded
    as the base; `model` (the command line's name for it) isn't needed."""
    os.makedirs(dest, exist_ok=True)
    rank = max(A.shape[0] for A, _ in factors.values())
    tensors = {}
    for mod, (A, B) in factors.items():
        key = mod if mod.startswith(_PEFT_PREFIX) else _PEFT_PREFIX + mod
        tensors[key + ".lora_A.weight"], tensors[key + ".lora_B.weight"] = A, B
    st_save_file(tensors, os.path.join(dest, "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "r": rank, "lora_alpha": rank, "lora_dropout": 0.0, "bias": "none",
           "target_modules": sorted({mod.rsplit(".", 1)[-1] for mod in factors}),
           "task_type": "CAUSAL_LM", "base_model_name_or_path": model_id}
    with open(os.path.join(dest, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return dest, rank


def _read_base(path: str) -> tuple[str, int, dict[str, tuple[int, int]], int]:
    """(architecture, block count, {tensor name: (out, in)}, attention head count) from the GGUF's header.
    Only the header is parsed; the weights are never read (GGUFReader memory-maps lazily)."""
    from gguf import GGUFReader
    r = GGUFReader(path)
    arch = r.fields["general.architecture"].contents()
    n_blocks = int(r.fields[f"{arch}.block_count"].contents())
    heads = r.fields.get(f"{arch}.attention.head_count")
    # GGUF lists dims innermost-first (ne), i.e. reversed from the (out, in) a weight has in HF.
    shapes = {t.name: tuple(int(x) for x in reversed(list(t.shape))) for t in r.tensors}
    return arch, n_blocks, shapes, int(heads.contents()) if heads is not None else 0


def _gguf_names(arch: str, n_blocks: int, modules) -> dict[str, str]:
    """{module: the base GGUF's tensor name} via gguf-py's per-architecture map (the same one
    convert_hf_to_gguf uses), so any architecture it knows is covered without a list of our own."""
    import gguf
    from gguf.tensor_mapping import get_tensor_name_map
    enum = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}.get(arch)
    if enum is None:
        return {}
    tm = get_tensor_name_map(enum, n_blocks)
    out = {}
    for mod in modules:
        parts = _hf_path(mod).split(".")
        # The map knows a bare decoder's names (`model.layers.N…`); a multimodal checkpoint nests them
        # (`model.language_model.layers.N…`), which the converter handles by stripping the outer prefix —
        # so try every suffix of the path, longest first.
        for i in range(len(parts)):
            name = tm.get_name(".".join(parts[i:]) + ".weight", try_suffixes=(".weight",))
            if name:
                out[mod] = name
                break
    return out


def _permute(B: np.ndarray, n_head: int) -> np.ndarray:
    # convert_hf_to_gguf's LlamaModel interleaves q/k rows for its RoPE layout, and convert_lora_to_gguf
    # applies the same to lora_b (rows = out) — so must we, or the adapter would land on scrambled rows.
    return B.reshape(n_head, 2, B.shape[0] // n_head // 2, B.shape[1]).swapaxes(1, 2).reshape(B.shape)


def build_gguf(factors: dict, base_path: str, dest: str, model_id: str = "",
               out=sys.stderr) -> tuple[str, int] | None:
    """A llama.cpp LoRA adapter at `dest`.gguf (what `--lora` loads) for the GGUF at `base_path` (`-m`; the
    federation's `model_id` isn't needed, the base file says it all): each module's factors
    under the base tensor's own name (`blk.N.attn_q.weight.lora_a` / `.lora_b`), checked against the
    base's shapes so a global for another model is skipped rather than handed to llama-server, which
    refuses to start on a mismatched adapter. alpha is written as 0, which llama.cpp reads as "scale 1"
    whatever the rank — the scale is already in B. None when nothing matched."""
    import gguf
    if not os.path.isfile(base_path):
        print(f"roger: {base_path} isn't a local file, so the federation's update can't be attached "
              "(point -m at a local gguf).", file=out)
        return None
    arch, n_blocks, shapes, n_head = _read_base(base_path)
    names = _gguf_names(arch, n_blocks, factors)
    tensors, rank = [], 0
    for mod, (A, B) in factors.items():
        name = names.get(mod)
        if name is None or shapes.get(name) != (B.shape[0], A.shape[1]):
            print(f"roger: federation update has no matching tensor for {_hf_path(mod)} in "
                  f"{os.path.basename(base_path)}; skipping it", file=out)
            continue
        if arch == "llama" and n_head and name.endswith("attn_q.weight"):
            B = _permute(B, n_head)
        tensors += [(name + ".lora_a", A), (name + ".lora_b", B)]
        rank = max(rank, A.shape[0])
    if not tensors:
        return None
    path = dest + ".gguf"
    w = gguf.GGUFWriter(path + ".tmp", arch)                 # arch must equal the base's or llama.cpp refuses
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, 0.0)
    for name, t in tensors:
        w.add_tensor(name, t.astype(np.float16))              # the converter's default precision
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(path + ".tmp", path)                           # a half-written adapter is never on disk
    return path, rank


# ---------------------------------------------------------------------------
# The runtimes
# ---------------------------------------------------------------------------

# The name under which vllm serves the federation adapter (`--lora-modules NAME=dir`); requests must ask
# for it by name, which the proxy does on the client's behalf so nothing changes on their side.
LORA_NAME = "roger"

# vllm only accepts these for --max-lora-rank (its LoRA kernels are compiled per rank).
_VLLM_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def _vllm_rank(r: int) -> int:
    return next((n for n in _VLLM_RANKS if n >= r), _VLLM_RANKS[-1])


# Per-runtime knowledge, keyed by the runtime binary's basename — the table of everything roger touches
# beyond `--port`. `model_flags`/`model_after`: where the served model is named (flags, and/or the
# positional after a subcommand) — what the daily pull resolves against the federation's allowlist and
# what the adapter is built for. `build`: writes the adapter in the format the runtime loads a LoRA in
# (signature of build_gguf/build_peft). `adapter_args`: the arguments that attach one at `path` of rank `r`.
# `adapter_flags`: the same territory seen from the user's side — the flags that mean they are already
# loading a LoRA of their own, which roger must not fight over (see `user_adapter`).
# `request_model`: the model name chat requests must carry to hit the adapter (llama-server applies
# `--lora` to everything; vllm serves it as a separate model).
RUNTIMES = {
    "llama-server": dict(
        model_flags=("-m", "--model"), model_after=None,
        build=build_gguf, adapter_args=lambda path, r: ["--lora", path], request_model=None,
        adapter_flags=("--lora", "--lora-scaled")),
    "vllm": dict(
        model_flags=("--model",), model_after="serve",
        build=build_peft,
        adapter_args=lambda path, r: ["--enable-lora", "--lora-modules", f"{LORA_NAME}={path}",
                                      "--max-lora-rank", str(_vllm_rank(r))],
        request_model=LORA_NAME,
        adapter_flags=("--enable-lora", "--lora-modules", "--max-lora-rank")),
}


def runtime_name(argv0: str) -> str:
    return os.path.basename(argv0).removesuffix(".exe")


def served_model(argv: list[str]) -> str | None:
    """The model as named on the runtime's command line (a gguf path for llama-server, an HF id for vllm),
    per RUNTIMES; None for an unknown runtime or when the command line doesn't name one (llama-server's
    `-hf` download, a vllm config file, …)."""
    spec = RUNTIMES.get(runtime_name(argv[0]))
    if spec is None:
        return None
    for flag in spec["model_flags"]:
        if (v := _flag_value(argv, flag)) is not None:
            return v
    if spec["model_after"] and spec["model_after"] in argv:
        i = argv.index(spec["model_after"]) + 1
        if i < len(argv) and not argv[i].startswith("-"):
            return argv[i]
    return None


def names_model(spec: dict) -> str:
    """How this runtime is told which model to serve ("-m / --model"), for the warning when the command
    line doesn't — derived from the same table `served_model` reads, so it can't drift from it."""
    ways = list(spec["model_flags"])
    if spec["model_after"]:
        ways.insert(0, f"{spec['model_after']} <model>")
    return " / ".join(ways)


def user_adapter(argv: list[str]) -> str | None:
    """The LoRA flag the user put on the command line themselves, or None. roger attaches the federation's
    update through those very flags, so appending its own would either stack two adapters (llama-server
    applies every `--lora`) or overwrite the user's (a repeated `--lora-modules`/`--max-lora-rank` is
    last-wins) — in both cases silently serving something neither side asked for. Their adapter wins."""
    spec = RUNTIMES.get(runtime_name(argv[0]))
    if spec is None:
        return None
    for a in argv[1:]:
        if a.split("=", 1)[0] in spec["adapter_flags"]:       # both `--lora X` and `--lora=X`
            return a.split("=", 1)[0]
    return None


def attach_adapter(p: Plan, argv0: str, path: str, rank: int) -> Plan:
    """The plan with the runtime's LoRA arguments appended (and, where the adapter is served under its own
    name, the request-model rewrite switched on). Unchanged for a runtime RUNTIMES doesn't know."""
    spec = RUNTIMES.get(runtime_name(argv0))
    if spec is None:
        return p
    return p._replace(child_argv=p.child_argv + spec["adapter_args"](path, rank),
                      request_model=spec["request_model"])
