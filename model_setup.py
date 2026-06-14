from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from huggingface_hub import get_safetensors_metadata
import os, torch

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

def find_think_delims(tokenizer) -> tuple[str, str] | None:
    """Probe the chat template for the thinking/reasoning channel delimiters.

    Forces a message with reasoning_content through the template; the first new special
    token (before the tool-call open) is the channel open, the next is its close.
    Returns decoded strings (e.g. ("<|channel>", "<channel|>") for Gemma-4) so the text
    renderer can do string matching without needing token IDs.
    Returns None if the template renders no thinking channel (non-reasoning model).
    """
    forcing  = [{"role": "user", "content": "x"},
                {"role": "assistant", "reasoning_content": "r",
                 "tool_calls": [{"id": "0", "type": "function",
                                 "function": {"name": "f", "arguments": {}}}]}]
    baseline = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "x"}]
    result = _probe_delims(tokenizer, forcing, baseline)
    if result is None:
        return None
    open_id, close_id = result
    if close_id is None:
        return None     # can't delimit without a close token; stream raw
    return tokenizer.decode([open_id]), tokenizer.decode([close_id])

def find_gen_prompt(tokenizer) -> list[int]:
    """Token ids of the assistant-turn cue (add_generation_prompt diff). Returns [] if none."""
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": "_"}], tokenize=True, add_generation_prompt=False)
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": "_"}], tokenize=True, add_generation_prompt=True)
    base = base["input_ids"] if isinstance(base, dict) else base
    full = full["input_ids"] if isinstance(full, dict) else full
    return full[len(base):]

def find_tool_res_id(tokenizer) -> int:
    """Last token of a dummy tool-call assistant turn — the tool_response boundary token."""
    out = tokenizer.apply_chat_template(
        [{"role": "assistant", "tool_calls": [
            {"id": "0", "type": "function", "function": {"name": "_", "arguments": {}}}]}],
        tokenize=True, add_generation_prompt=False)
    ids = out["input_ids"] if isinstance(out, dict) else out
    return ids[-1]

def _param_count(model_id: str) -> int | None:
    """Total parameter count from HF safetensors metadata (no weight download). None on failure."""
    try:
        return sum(get_safetensors_metadata(model_id).parameter_count.values())
    except Exception:
        return None    # offline / gated / no safetensors → caller falls back to 4-bit

def _4bit(compute_dtype) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype, llm_int8_enable_fp32_cpu_offload=True)

def _select_quant(model_id, gpu_available, for_training):
    """Pick the highest-precision tier that fits VRAM: (quantization_config | None, dtype).

    Estimates the model footprint from its param count and compares each tier
    (bf16 → int8 → nf4) against ~VRAM_UTIL of total VRAM, inflated by TRAIN_HEADROOM when
    loading for RL. The 4-bit tier is also the fallback; any overflow is offloaded by
    device_map="auto". bitsandbytes needs CUDA, so the no-GPU path loads unquantized.
    """
    if not gpu_available:
        return None, torch.float32
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    P = _param_count(model_id)
    if P is None:
        return _4bit(compute_dtype), compute_dtype     # safe default when size is unknown
    budget = sum(torch.cuda.get_device_properties(i).total_memory
                 for i in range(torch.cuda.device_count())) * VRAM_UTIL
    factor = TRAIN_HEADROOM if for_training else 1.0
    if 2.0 * P * factor <= budget:                     # bf16 weights fit → no quant
        return None, compute_dtype
    if 1.0 * P * factor <= budget:                     # int8 fits
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True), compute_dtype
    return _4bit(compute_dtype), compute_dtype          # nf4 (offload remainder if still tight)

def fetch_model(model_id="google/gemma-4-E2B-it", for_training: bool = False) -> tuple[AutoModelForImageTextToText, AutoProcessor, tuple[int, int]]:
    # VRAM-aware quantization: choose tier from model size vs available VRAM
    gpu_available = torch.cuda.is_available()
    quant_cfg, dtype = _select_quant(model_id, gpu_available, for_training)
    # device_map="auto" shards across GPUs and offloads overflow to CPU/disk (offload_folder)
    offload_dir = os.path.join(os.getcwd(), ".roger", "scratch", "offload")
    os.makedirs(offload_dir, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_buffers=True,
        offload_folder=offload_dir,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    # Find tool call tokens
    tool_tokens = find_tool_call_tokens(processor.tokenizer)
    return model, processor, tool_tokens