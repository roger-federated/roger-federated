# Check whether gpu is available,
# Then check whether GPU-torch is installed, along with bitsandbytes, transformers, and accelerate
# Inspect CPU/GPU memory capacity and precision, and setup the quantization/offloading/compute_dtype config accordingly
# If sufficient GPU memory, use vLLM for inference

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

def _probe_delims(tokenizer, forcing, baseline):
    """Return (open_id, close_id) of the first token `forcing` adds over `baseline` (special-tokens only)."""
    def _tok(msgs):
        out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        return out["input_ids"] if isinstance(out, dict) else out
    sp   = set(tokenizer.all_special_ids)
    base = {t for t in _tok(baseline) if t in sp}
    seq  = [t for t in _tok(forcing)  if t in sp]
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

def fetch_model(model_id="google/gemma-4-E2B-it") -> tuple[AutoModelForImageTextToText, AutoProcessor, tuple[int, int]]:
    # Quantization
    gpu_available = torch.cuda.is_available()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if (gpu_available and torch.cuda.is_bf16_supported()) else torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_quant_type="nf4"
    )
    # Load model on GPUs if available
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": i for i in range(torch.cuda.device_count())} if gpu_available else "auto",
        low_cpu_mem_usage=True,
        offload_buffers=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    # Find tool call tokens
    tool_tokens = find_tool_call_tokens(processor.tokenizer)
    return model, processor, tool_tokens