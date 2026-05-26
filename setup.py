# Check whether gpu is available, 
# Then check whether GPU-torch is installed, along with bitsandbytes, transformers, and accelerate
# Inspect CPU/GPU memory capacity and precision, and setup the quantization/offloading/compute_dtype config accordingly
# If sufficient GPU memory, use vLLM for inference

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch

def fetch_model(model_id="google/gemma-4-E2B-it"):
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
    return model, processor

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