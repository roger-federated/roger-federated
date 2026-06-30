from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMultimodalLM, BitsAndBytesConfig
from huggingface_hub import get_safetensors_metadata
import os, torch
import psutil
from roger.agency.path_utils import state_dir

VRAM_UTIL     = 0.9   # fraction of total VRAM we aim to fill; the rest absorbs activations/KV
TRAIN_HEADROOM = 1.7  # inflate the footprint estimate when loading for RL (grads + optimizer)

def _probe_delims(tokenizer, forcing, baseline):
    """Return (open_id, close_id) of the first token `forcing` adds over `baseline` (special-tokens only)."""
    sp   = set(tokenizer.all_special_ids)
    base = {t for t in tokenizer.apply_chat_template(
        baseline, tokenize=True, add_generation_prompt=False
        )["input_ids"] if t in sp}
    seq  = [t for t in tokenizer.apply_chat_template(
        forcing, tokenize=True, add_generation_prompt=False
        )["input_ids"] if t in sp]
    for i, t in enumerate(seq):
        if t not in base:
            return t, (seq[i + 1] if i + 1 < len(seq) else None)
    return None

def find_tool_call_tokens(tokenizer) -> tuple[int, int]:
    """Probe the chat template to find the token-pair that brackets a tool call.

    Forces a tool-call message through the template; the first special token that is
    absent from a plain assistant message is the open delimiter, the next is the close.
    This is robust across models (Gemma-4, Llama-3, Qwen, …) without a hardcoded list.
    Raises ValueError if the model does not support tool calls (no open found) or omits
    the close token (Mistral-style; unsupported by the current rollout loop).
    """
    forcing  = [{"role": "user", "content": "x"},
                {"role": "assistant", "tool_calls": [
                    {"id": "0", "type": "function",
                     "function": {"name": "f", "arguments": {}}}]}]
    baseline = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "x"}]
    result = _probe_delims(tokenizer, forcing, baseline)
    if result is None:
        raise ValueError("find_tool_call_tokens: chat template emits no new special tokens for a tool call — model may not support tool use.")
    open_id, close_id = result
    if close_id is None:
        raise ValueError("find_tool_call_tokens: no close-delimiter found (Mistral-style single-token calls are unsupported).")
    return open_id, close_id

def find_think_tokens(tokenizer) -> tuple[int, int] | None:
    """Probe the chat template for the thinking/reasoning channel delimiter token ids.

    Forces a message with reasoning_content through the template; the first new special
    token (before the tool-call open) is the channel open, the next is its close. Mirrors
    find_tool_call_tokens but is non-fatal. Callers needing strings decode the ids (e.g.
    the text renderer); the rollout injects the open id to seed a thought.
    Returns None if the template renders no thinking channel (non-reasoning model) or omits
    a close token (can't delimit without it; stream raw).
    """
    forcing  = [{"role": "user", "content": "x"},
                {"role": "assistant", "reasoning_content": "r",
                 "tool_calls": [{"id": "0", "type": "function",
                                 "function": {"name": "f", "arguments": {}}}]}]
    baseline = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "x"}]
    result = _probe_delims(tokenizer, forcing, baseline)
    if result is None or result[1] is None:
        return None
    return result

def find_gen_prompt(tokenizer) -> list[int]:
    """Token ids of the assistant-turn cue (add_generation_prompt diff). Returns [] if none."""
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": "_"}], tokenize=True, add_generation_prompt=False)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": "_"}], tokenize=True, add_generation_prompt=True)
    return full["input_ids"][len(base["input_ids"]):]

def find_tool_res_id(tokenizer) -> int:
    """Last token of a dummy tool-call assistant turn — the tool_response boundary token."""
    out = tokenizer.apply_chat_template(
        [{"role": "assistant", "tool_calls": [
            {"id": "0", "type": "function", "function": {"name": "_", "arguments": {}}}]}],
        tokenize=True, add_generation_prompt=False)
    return out["input_ids"][-1]

def _param_count(model_id: str) -> int | None:
    """Total parameter count from HF safetensors metadata (no weight download). None on failure."""
    try:
        return sum(get_safetensors_metadata(model_id).parameter_count.values())
    except Exception:
        return None    # offline / gated / no safetensors → caller falls back to 4-bit

def _vram_budget() -> dict[int, int]:
    """Per-GPU usable byte budget from *free* (not total) VRAM.

    Free memory is what Accelerate can actually place weights into; sizing tiers against the
    same number we hand Accelerate as max_memory keeps the two from disagreeing.
    """
    return {i: int(torch.cuda.mem_get_info(i)[0] * VRAM_UTIL)
            for i in range(torch.cuda.device_count())}

def _4bit(compute_dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype, llm_int8_enable_fp32_cpu_offload=True)

def _select_quant(model_id, gpu_available, for_training): 
    """Pick the highest-precision tier that fits VRAM: (quantization_config | None, dtype, fits).

    Estimates the model footprint from its param count and compares each tier
    (bf16 → int8 → nf4) against ~VRAM_UTIL of *free* VRAM, inflated by TRAIN_HEADROOM when
    loading for RL. `fits` is True when the chosen tier's weights sit within that budget, so
    the caller can pin the whole model to the GPU; it's False for the unknown-size fallback
    and the 4-bit-still-too-big case, where Accelerate must place + offload the remainder.
    bitsandbytes needs CUDA, so the no-GPU path loads unquantized.
    """
    if not gpu_available:
        return None, torch.float32, False
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    P = _param_count(model_id)
    if P is None:
        return _4bit(compute_dtype), compute_dtype, False   # unknown size → let auto offload
    budget = sum(_vram_budget().values())              # free VRAM (matches the pin/offload budget)
    factor = TRAIN_HEADROOM if for_training else 1.0
    if 2.0 * P * factor <= budget:                     # bf16 weights fit → no quant
        return None, compute_dtype, True
    if 1.0 * P * factor <= budget:                     # int8 fits
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True), compute_dtype, True
    return _4bit(compute_dtype), compute_dtype, 0.5 * P * factor <= budget   # nf4; fits unless still too big

def placement_summary(model) -> tuple[str, str]:
    """Where did the weights land? Returns (message, rich_style), derived from hf_device_map.
    CPU/disk placement means no/partial GPU acceleration, so flag it as a warning."""
    devmap = getattr(model, "hf_device_map", None)
    if not devmap:                             # no device_map (e.g. plain .to(device)) → probe a param
        try:
            dev = str(next(model.parameters()).device)
        except StopIteration:
            return "Model loaded.", "green"     # no params to probe; stay silent on placement
        return ("Model loaded; fully on GPU.", "green") if dev.startswith("cuda") \
            else ("Model loaded on CPU; no GPU acceleration (slow).", "yellow")
    devices = set(devmap.values())
    on_gpu  = any(isinstance(d, int) or (isinstance(d, str) and d.startswith("cuda"))
                  for d in devices)
    on_cpu  = "cpu"  in devices
    on_disk = "disk" in devices
    if on_gpu and (on_cpu or on_disk):
        where = "disk" if on_disk else "CPU"
        return f"Model loaded; partly offloaded to {where} (slower; weights didn't all fit in VRAM).", "yellow"
    if on_cpu or on_disk:                      # no GPU at all
        return "Model loaded on CPU; no GPU acceleration (slow).", "yellow"
    return "Model loaded; fully on GPU.", "green"

def _placement_kwargs(quant_cfg, dtype, gpu_available, n_gpu, fits) -> dict:
    """The from_pretrained kwargs for the chosen tier + placement (shared by the normal and the
    folded temp-dir reload path)."""
    kwargs = dict(quantization_config=quant_cfg, dtype=dtype, low_cpu_mem_usage=True,
                  attn_implementation="sdpa")
    if not gpu_available:                # CPU
        kwargs["device_map"] = "auto"
    elif n_gpu == 1 and fits:            # Fits on one GPU (device_map="auto" is unreliable; pin it)
        kwargs["device_map"] = {"": 0}
    elif fits:                           # Fits on multiple GPUs
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = _vram_budget()
    else:                                # Doesn't fit on GPU → offload the remainder
        offload_dir = os.path.join(state_dir(), "scratch", "offload")
        os.makedirs(offload_dir, exist_ok=True)
        kwargs["device_map"]      = "auto"
        kwargs["max_memory"]      = {**_vram_budget(), "cpu": psutil.virtual_memory().available}
        kwargs["offload_buffers"] = True
        kwargs["offload_folder"]  = offload_dir
    return kwargs


def _quantize_inplace(model, quant_cfg) -> None:
    """Convert every nn.Linear (minus the modules transformers keeps in full precision, e.g. lm_head)
    into its bnb counterpart in place. The new Params hold the *current* (already ΔW-folded) bf16
    weight unquantized; bnb quantizes them when the caller moves the model to CUDA. Verified to match
    `from_pretrained(folded_bf16, quantization_config)` exactly — this is the no-disk fold path."""
    import torch.nn as nn
    import bitsandbytes as bnb
    try:                                              # moved across transformers versions
        from transformers.quantizers.base import get_keys_to_not_convert
    except ImportError:
        from transformers.integrations.bitsandbytes import get_keys_to_not_convert
    skip = set()
    for key in get_keys_to_not_convert(model):
        try:
            skip.add(model.get_submodule(key))
        except AttributeError:
            pass
    is4 = getattr(quant_cfg, "load_in_4bit", False)

    def convert(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child not in skip:
                if is4:
                    new = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                            bias=child.bias is not None,
                                            compute_dtype=quant_cfg.bnb_4bit_compute_dtype,
                                            quant_type=quant_cfg.bnb_4bit_quant_type,
                                            compress_statistics=quant_cfg.bnb_4bit_use_double_quant)
                    new.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False,
                                                   quant_type=quant_cfg.bnb_4bit_quant_type,
                                                   compress_statistics=quant_cfg.bnb_4bit_use_double_quant)
                else:
                    new = bnb.nn.Linear8bitLt(child.in_features, child.out_features,
                                              bias=child.bias is not None, has_fp16_weights=False)
                    new.weight = bnb.nn.Int8Params(child.weight.data, requires_grad=False,
                                                   has_fp16_weights=False)
                if child.bias is not None:
                    new.bias = nn.Parameter(child.bias.data, requires_grad=False)
                setattr(parent, name, new)
            else:
                convert(child)

    convert(model)
    model.is_loaded_in_4bit, model.is_loaded_in_8bit = is4, not is4


def _load_folded(model_id, deltas, model_cls, quant_cfg, dtype, fits, gpu_available, n_gpu):
    """Load the base in bf16, fold the federated global ΔW into the *float* weights, then quantize +
    place. The HF cache is never altered and no model is persisted. The common single-GPU-fits case
    quantizes in memory (no disk); offload/multi-GPU fall back to a transient temp-dir reload."""
    from roger.federated import delta as delta_mod
    model = model_cls.from_pretrained(model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      attn_implementation="sdpa")
    delta_mod.fold_into(model, deltas)
    if not gpu_available:
        return model                                       # CPU, bf16
    if fits and n_gpu == 1:                                 # in-memory quantize/place — no disk
        if quant_cfg is not None:
            _quantize_inplace(model, quant_cfg)
        return model.to("cuda:0")
    # Offload / multi-GPU: round-trip the folded bf16 weights through a temp dir so from_pretrained's
    # device_map/offload machinery handles placement. Transient (deleted), not a persisted model.
    import tempfile, shutil
    tmp = tempfile.mkdtemp(prefix="roger_fold_")
    try:
        model.save_pretrained(tmp)
        del model
        return model_cls.from_pretrained(tmp, **_placement_kwargs(quant_cfg, dtype, gpu_available, n_gpu, fits))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _resolve_model_cls(model_id: str):
    """Pick the right Auto model class from the HF config registry (no weights download).
    Priority: MultimodalLM (any-to-any) > ImageTextToText > CausalLM."""
    try:
        cfg = AutoConfig.from_pretrained(model_id)
    except Exception:
        return AutoModelForCausalLM
    for cls in [AutoModelForMultimodalLM, AutoModelForImageTextToText, AutoModelForCausalLM]:
        if type(cfg) in cls._model_mapping:
            return cls
    return AutoModelForCausalLM


def fetch_model(model_id="google/gemma-4-E2B-it", for_training: bool = False,
                model_cls=None,
                weight_deltas: dict | None = None) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    if model_cls is None:
        model_cls = _resolve_model_cls(model_id)
    # `model_cls` lets non-generative models reuse this loader (e.g. the privacy filter's
    # AutoModelForTokenClassification) so they get the same VRAM-aware quant/placement.
    # `weight_deltas` (federated): a dense ΔW per module folded into the base before quantization.
    # VRAM-aware quantization: choose tier from model size vs available VRAM
    gpu_available = torch.cuda.is_available()
    quant_cfg, dtype, fits = _select_quant(model_id, gpu_available, for_training)
    n_gpu = torch.cuda.device_count() if gpu_available else 0
    if weight_deltas:
        model = _load_folded(model_id, weight_deltas, model_cls, quant_cfg, dtype, fits, gpu_available, n_gpu)
    else:
        model = model_cls.from_pretrained(
            model_id, **_placement_kwargs(quant_cfg, dtype, gpu_available, n_gpu, fits))
    # Text-only classifiers (e.g. token-classification) ship no processor config; fall back to the
    # plain tokenizer so callers always get a usable text front-end.
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except (ValueError, OSError, KeyError):
        processor = AutoTokenizer.from_pretrained(model_id)
    return model, processor


def load_drafter(draft_id: str, target_tokenizer):
    """Load a speculative-decoding draft model, or None if it doesn't share the target's vocab.

    Assisted generation needs a shared token<->id map; we compare vocabs by loading only the draft
    tokenizer (no model download), so an incompatible drafter is rejected before any heavy load.
    """
    from transformers import AutoTokenizer
    if AutoTokenizer.from_pretrained(draft_id).get_vocab() != target_tokenizer.get_vocab():
        return None
    return fetch_model(draft_id, model_cls=AutoModelForCausalLM)[0]


def uses_sliding_window(model) -> bool:
    cfg = model.config
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    if not getattr(cfg, "sliding_window", None):
        return False
    lt = getattr(cfg, "layer_types", None)
    return lt is None or any("sliding" in str(t) for t in lt)