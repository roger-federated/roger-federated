"""lora_utils.py — attach the LoRA adapter a training round updates.

The adapter is the federation's global (training/wire_trainer.py initialises its factors from the
pulled blob, or from the derived cold start), so a round starts exactly where the federation is and
what it shares is the epoch's factor Δ against that start. Nothing is persisted locally: the model
only changes when the next global is pulled and rebuilt as the runtime's adapter (runtime/adapter.py).
"""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Fixed federation LoRA basis (the shared secure-agg layout; masks only cancel if every member targets
# the same modules, in the same sorted-key order). Not "all-linear": a broad set bloats the server's
# per-round I/O for little gain. q/v ≈ 6% of all-linear; lm_head/embeddings excluded (vocab-sized).
FED_TARGETS = ["q_proj", "v_proj"]


def attach_lora(model, *, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                targets=FED_TARGETS,                 # all training is federated → default to the basis
                rank_pattern: dict | None = None):
    """Return the model wrapped with a single trainable LoRA adapter. Works for a quantized (QLoRA)
    or full base. `rank_pattern` {module: rank}: the federation's per-module rank map
    (federated/delta.rank_for); alpha follows it, so every module's scale is 1 — the factor contract
    keeps the scale inside the factors."""
    is_quantized = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    if is_quantized:
        # casts norms to fp32, enables input grads, turns on gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()   # checkpointing needs a grad-bearing input on a frozen base
    cfg = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=targets, bias="none",   # targets: name list or "all-linear"
                     rank_pattern=rank_pattern or {}, alpha_pattern=rank_pattern or {})
    model = get_peft_model(model, cfg)
    model.config.use_cache = False   # incompatible with gradient checkpointing
    return model
