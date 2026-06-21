"""lora_utils.py — LoRA adapter attach (for RL training) and load (for serving).

`target_modules` is just the selection of which existing layers get the LoRA branch. 
The trained adapter lives at ~/.roger/adapter and is resumed in place across sessions, 
which is what makes the local RL genuinely continual.
"""
import os

from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training,
                  PeftModel)

from roger.agency.path_utils import state_dir

ADAPTER_DIR = os.path.join(state_dir(), "adapter")   # ~/.roger/adapter


def adapter_exists() -> bool:
    # A saved PEFT adapter always carries its config alongside the weights.
    return os.path.isdir(ADAPTER_DIR) and os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json"))


def attach_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                targets="all-linear"):
    """Return a trainable PEFT-wrapped model. Resumes ~/.roger/adapter when present so updates
    accumulate; otherwise starts a fresh adapter. Works for a quantized (QLoRA) or full base."""
    # Ctrl-D reuse: model already wears the inference adapter; re-injecting would nest wrappers,
    # so unfreeze the existing adapter in place instead.
    if isinstance(model, PeftModel):
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad_(True)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        return model

    is_quantized = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    if is_quantized:
        # casts norms to fp32, enables input grads, turns on gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # checkpointing needs a grad-bearing input on a frozen base
    if adapter_exists():
        model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=True)
    else:
        # 'all-linear' = every nn.Linear (minus head), model-agnostic; vision-tower linears get
        # wrapped too but receive no gradient on v1's text-only episodes.
        cfg = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=targets, bias="none")
        model = get_peft_model(model, cfg)
    model.config.use_cache = False   # incompatible with gradient checkpointing
    return model


def load_adapter_for_inference(model):
    """Return the base wrapped with the trained adapter for serving (new wrapper, so reassign), or
    the unchanged model when none exists."""
    if not adapter_exists():
        return model
    return PeftModel.from_pretrained(model, ADAPTER_DIR)
