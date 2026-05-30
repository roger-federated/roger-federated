# Check whether gpu is available, 
# Then check whether GPU-torch is installed, along with bitsandbytes, transformers, and accelerate
# Inspect CPU/GPU memory capacity and precision, and setup the quantization/offloading/compute_dtype config accordingly
# If sufficient GPU memory, use vLLM for inference

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

def find_tool_call_tokens(tokenizer):
    # Common tool call tokens
    tool_call_tokens = [
        ("<|tool_call>", "<tool_call|>"),
        ("<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"),
        ("<|python_tag|>", "<|eom_id|>"),
        ("<|tool_call_begin|>", "<|tool_call_end|>"),
    ]
    # Find which pair of tool call delimiters this model's tokenizer knows as single tokens
    tool_tokens = None
    for start, end in tool_call_tokens:
        s = tokenizer.encode(start, add_special_tokens=False)
        e = tokenizer.encode(end, add_special_tokens=False)
        if len(s) == 1 and len(e) == 1: # TODO: does not support Mistral's omittance of end token
            tool_tokens = (s[0], e[0])
            return tool_tokens
    else:
        raise ValueError("Tokenizer does not encode any of the known tool call tokens.")

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
    # Add new special tokens for state representation
    init_new_tokens(["<|state>", "<state|>"], ["environment", "observation", "start", "end"], torch.tensor([[.3,.3,.4,0.],[.3,.3,0.,.4]]), model, processor.tokenizer)
    # Find tool call tokens
    tool_tokens = find_tool_call_tokens(processor.tokenizer)
    return model, processor, tool_tokens

def init_new_tokens(new_tokens, like_tokens, weights, model, tokenizer):
    """
    Args:
        new_tokens: list of new token strings to add to the tokenizer
        like_tokens: list of existing token strings to base the new token embeddings on
        weights: tensor of shape (len(new_tokens), len(like_tokens)) specifying the weights for combining the like_token embeddings to create the new token embeddings
    Returns:
        list of new token ids corresponding to the new_tokens
    """
    # Find embeddings of related tokens
    token_ids = tokenizer.encode(like_tokens)
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(torch.tensor(token_ids, device=embed_layer.weight.device)).squeeze(1)
    # Merge them into new similar tokens
    assert weights.shape == (len(new_tokens), len(like_tokens))
    assert torch.allclose(weights.sum(axis=1), torch.ones(len(new_tokens)))
    embeds = (embeds * weights.to(embeds.device, embeds.dtype).unsqueeze(-1)).sum(axis=1)
    # Insert into tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        embed_layer.weight.data[-len(new_tokens):] = embeds
    return tokenizer.encode(new_tokens)