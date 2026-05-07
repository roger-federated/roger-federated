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