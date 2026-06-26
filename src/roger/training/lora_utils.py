"""lora_utils.py — attach a fresh LoRA adapter for an RL training round.

A round trains a brand-new adapter on top of whatever base is loaded (the HF base with the federated
global already folded into its weights, see federated/delta.py). Because LoRA inits B=0, the adapter
starts at ΔW=0, so after the REINFORCE++ step its ΔW = (alpha/r)·B@A is exactly the local update we
densify and share. The adapter is never persisted: the local model only changes when the next global
is pulled and folded at load time, so there is no inference-time adapter and no resume-in-place.
"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def attach_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                targets="all-linear"):
    """Return the model wrapped with a single trainable LoRA adapter. Works for a quantized (QLoRA)
    or full base. The Ctrl-D reuse path passes the already-loaded (non-PeftModel) served model."""
    is_quantized = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    if is_quantized:
        # casts norms to fp32, enables input grads, turns on gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # checkpointing needs a grad-bearing input on a frozen base
    # 'all-linear' = every nn.Linear (minus head), model-agnostic.
    cfg = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=targets, bias="none")
    model = get_peft_model(model, cfg)
    model.config.use_cache = False   # incompatible with gradient checkpointing
    return model


def local_state_dict(model):
    """The trained adapter's LoRA factors as the canonical PEFT state dict — keys stripped of the
    adapter name, ready to densify into the ΔW we share."""
    from peft import get_peft_model_state_dict
    return get_peft_model_state_dict(model)
