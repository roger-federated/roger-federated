"""wire_trainer.py — the REINFORCE++ round over the chats the runtime wrapper captured.

trainer.py holds the REINFORCE++ step itself and wants token-level episodes; all roger has is the
OpenAI-wire JSON under ~/.roger/messages/ (runtime/capture.py), each graded by the model itself
(`self_eval`, runtime/grader.py) and scored from its tool results (`tool_signals`, runtime/signals.py).
So an episode is rebuilt here: the conversation is re-rendered with the model's own
chat template, and each assistant turn's generated span is found by a prefix probe — template(turns before
it, generation prompt) vs template(through it) — with no template tags hardcoded. The behaviour log-probs
aren't on the wire (the runtime sampled them), so they are recomputed in one no-grad pass over the same
weights, which is what makes the update the on-policy REINFORCE++ step (ratio 1 at the first epoch).

Training happens with the federation's global adapter attached — in fact the global IS the adapter being
trained. Under the factor contract (federated/delta.py) an epoch trains one factor against the other,
frozen: the adapter is initialised from the persisted global (or, on a cold federation, the derived
`init_A` and B=0), only `phase`'s factor gets gradients, and what is shared is that factor's Δ. The model
is loaded without any download: llama-server's GGUF is dequantized by transformers (`gguf_file=`), and
vllm's HF weights come from the cache `vllm serve` filled (`local_files_only`).
"""
import json, os, re

import torch

from roger.federated import delta as delta_mod
from roger.training import lora_utils, trainer


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(source: str):
    """(model, tokenizer) for the model the runtime served: a local .gguf (llama-server's `-m`) is
    dequantized by transformers; anything else (vllm's HF id or local dir) is read from disk/cache only.
    Raises when transformers can't load it (e.g. an architecture its GGUF loader doesn't map yet)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if source.endswith(".gguf") and os.path.isfile(source):
        where, kw = os.path.dirname(os.path.abspath(source)), {"gguf_file": os.path.basename(source)}
    else:
        where, kw = source, {"local_files_only": True}
    tok = AutoTokenizer.from_pretrained(where, **kw)
    cuda = torch.cuda.is_available()
    load_kw = {"dtype": torch.bfloat16 if cuda or torch.backends.mps.is_available() else torch.float32}
    if cuda:
        load_kw["device_map"] = "auto"            # offloads to CPU when the model outgrows the GPU
        if "gguf_file" not in kw:                 # QLoRA on CUDA; a dequantized GGUF stays bf16
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
                load_kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            except ImportError:
                pass
    model = AutoModelForCausalLM.from_pretrained(where, **load_kw, **kw)
    if not cuda and torch.backends.mps.is_available():
        model = model.to("mps")
    return model, tok


def targets(model) -> dict[str, tuple[str, int, int]]:
    """{canonical module key: (module name in this model, out, in)} for the federation basis
    (lora_utils.FED_TARGETS) inside the text decoder only. The canonical key is `model.` + the path inside
    the decoder (`model.layers.N.self_attn.q_proj`), so a GGUF-loaded text-only model and a multimodal HF
    checkpoint (`model.language_model.layers.N…`) name the same module identically — one `compat` for the
    federation — and a vision tower's own q/v projections are never trained."""
    decoder = model.get_decoder()
    prefix = next(n for n, m in model.named_modules() if m is decoder)
    out = {}
    for name, mod in decoder.named_modules():
        if name.rsplit(".", 1)[-1] in lora_utils.FED_TARGETS and hasattr(mod, "in_features"):
            full = f"{prefix}.{name}" if prefix else name
            # out/in from the Linear's declared features: a bnb 4-bit weight is packed, its shape isn't.
            out["base_model.model.model." + name] = (full, mod.out_features, mod.in_features)
    return out


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

def _plain(content):
    """Text content, or None when a part isn't text (images/audio: a text-only forward would mis-embed)."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(p, dict) and p.get("type") == "text" for p in content):
        return "".join(p.get("text", "") for p in content)
    return None


def _messages(rec: dict) -> list | None:
    """The conversation as the template expects it: the request's history + the captured reply, with
    tool-call arguments decoded (the wire carries them as JSON strings, templates iterate them as maps)."""
    reply = rec["response"]["choices"][0]["message"]
    msgs = []
    for m in list(rec["request"].get("messages") or []) + [dict(reply, role="assistant")]:
        m = {k: v for k, v in m.items() if v is not None}
        if "content" in m:
            m["content"] = _plain(m["content"])
            if m["content"] is None:
                return None
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function") or {})
                try:
                    fn["arguments"] = json.loads(fn.get("arguments") or "{}")
                except (TypeError, ValueError):
                    pass                                  # already a map, or not JSON: leave as sent
                calls.append(dict(tc, function=fn))
            m["tool_calls"] = calls
        msgs.append(m)
    return msgs


def _ids(tok, msgs: list, tools, gen: bool) -> list[int]:
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gen)
    return tok(text, add_special_tokens=False)["input_ids"]


def episode(tok, rec: dict) -> dict | None:
    """{seq, traj: [{gen_start, masks}]} in the shape trainer._new_logps expects (masks all None: the
    runtime decoded unconstrained), one step per assistant turn whose span the prefix probe can locate. A turn is skipped
    when rendering it isn't a strict extension of the history (templates that rewrite earlier turns, e.g.
    dropping old reasoning), since its tokens then aren't what the model saw when it generated them."""
    msgs = _messages(rec)
    if msgs is None:
        return None
    tools = rec["request"].get("tools")
    full = _ids(tok, msgs, tools, gen=False)
    traj = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        pre, cur = _ids(tok, msgs[:i], tools, gen=True), _ids(tok, msgs[: i + 1], tools, gen=False)
        if cur[: len(pre)] != pre or full[: len(cur)] != cur or len(cur) <= len(pre):
            continue
        traj.append({"gen_start": len(pre), "masks": [None] * (len(cur) - len(pre))})
    if not traj:
        return None
    return {"dir": rec.get("_path", ""), "seq": torch.tensor(full), "traj": traj}


def reward(rec: dict) -> float:
    """Flat episode return: the self-evaluation's per-metric mean (the grader stores only the metrics;
    averaging them is the optimiser's call) + the mean of the user's reactions to the replies they
    answered (the human signal; a mean so a long chat doesn't outweigh a short one) + the verifiable
    tool-result signals of every step."""
    ev = rec["self_eval"]
    scores, reactions = ev["scores"], ev.get("reactions") or {}
    return (sum(scores.values()) / len(scores)
            + (sum(reactions.values()) / len(reactions) if reactions else 0.0)
            + sum(float(v) for v in (rec.get("tool_signals") or {}).values()))


# ---------------------------------------------------------------------------
# The round
# ---------------------------------------------------------------------------

def _factors(global_blob: bytes | None) -> dict:
    if global_blob is None:
        return {}
    tensors, _ = delta_mod.from_bytes(global_blob)
    return tensors


def train(records: list[dict], source: str, model_id: str, global_blob: bytes | None, status: dict, *,
          epochs: int = 1, lr: float = 1e-5, clip_eps: float = 0.2, max_grad_norm: float = 1.0) -> dict:
    """One REINFORCE++ round over `records` (graded capture files, each with `_path`) on the model at
    `source`, training `status["phase"]`'s factor of the federation's global (`global_blob`, or the derived
    cold start) at the federation's rank map. Returns {"trained": False, "reason"} or the upload:
    {update: {factor key: Δ}, base: {module: (out, in)}, epoch, n_episodes, …}."""
    epoch, phase, cap = int(status.get("epoch", 0)), status.get("phase", "B"), int(status.get("rank", 16))
    if phase != delta_mod.phase_of(epoch):
        return {"trained": False, "reason": f"federation reports phase {phase} for epoch {epoch}"}
    model, tok = load(source)

    eps, rets = [], []
    for rec in records:
        ep = episode(tok, rec)
        if ep is not None:
            eps.append(ep)
            rets.append(reward(rec))
    if len(eps) < 2:                                   # baseline + z-norm need a batch
        return {"trained": False, "reason": f"only {len(eps)} usable conversation(s)"}
    adv = trainer.advantages(torch.tensor(rets))
    if adv is None:
        return {"trained": False, "reason": "every conversation scored the same"}
    trainer.anonymize(eps, tok)
    if not any(float(ep["keep"].sum()) for ep in eps):
        return {"trained": False, "reason": "all generated tokens were PII"}

    mods = targets(model)
    base = {key: (o, i) for key, (_, o, i) in mods.items()}
    ranks = delta_mod.rank_map(base, cap)
    glob = _factors(global_blob)
    # No dropout: the behaviour log-probs below are the policy itself, and the ratio must start at 1.
    model = lora_utils.attach_lora(model, r=cap, alpha=cap, dropout=0.0,
                                   targets=[full for full, _, _ in mods.values()],
                                   rank_pattern={re.escape(mods[k][0]): r for k, r in ranks.items()})

    # The adapter starts AT the global: its factors are the federation's (or the derived cold start), and
    # only this epoch's factor is trainable. Its Δ against that start is the contribution.
    trained_key = delta_mod.SUFFIX[phase]
    start, params = {}, []
    for key, (full, out_dim, in_dim) in mods.items():
        lora = model.get_submodule("base_model.model." + full)
        r = ranks[key]
        A = glob.get(key + delta_mod.LORA_A)
        B = glob.get(key + delta_mod.LORA_B)
        if (A is None) != (B is None) or (A is not None and (tuple(A.shape) != (r, in_dim)
                                                              or tuple(B.shape) != (out_dim, r))):
            return {"trained": False, "reason": f"the federation's global doesn't fit this model at {key}"}
        if A is None:
            A, B = delta_mod.init_A(model_id, key, r, in_dim), torch.zeros(out_dim, r)
        wa, wb = lora.lora_A["default"].weight, lora.lora_B["default"].weight
        wa.data.copy_(A.to(wa.dtype))
        wb.data.copy_(B.to(wb.dtype))
        wa.requires_grad_(phase == "A")
        wb.requires_grad_(phase == "B")
        p = wa if phase == "A" else wb
        start[key + trained_key] = p.detach().float().cpu().clone()
        params.append((key + trained_key, p))

    # Behaviour log-probs: the runtime didn't expose them, so take them from the start-of-round weights
    # (== the served global) before any step — dropout off, so this is the policy itself.
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for ep in eps:
            for e in ep["traj"]:
                e["old_logp"] = torch.zeros(len(e["masks"]))
            lp, _ = trainer._new_logps(model, ep, device)
            off = 0
            for e in ep["traj"]:
                n = len(e["masks"])
                e["old_logp"] = lp[off: off + n].float().cpu()
                off += n
    model.train()
    trainable = [p for _, p in params]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(trainable, lr=lr) if device.type == "cuda" else None
    except ImportError:
        opt = None
    opt = opt or torch.optim.AdamW(trainable, lr=lr)
    loss = trainer.reinforce(model, eps, adv, opt, trainable, epochs=epochs, clip_eps=clip_eps,
                             max_grad_norm=max_grad_norm)
    update = {k: p.detach().float().cpu() - start[k] for k, p in params}
    return {"trained": True, "update": update, "base": base, "epoch": epoch, "phase": phase,
            "n_episodes": len(eps), "mean_return": float(sum(rets) / len(rets)), "loss": loss}
