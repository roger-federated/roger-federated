import torch, math
import transformers

class DummyEnv:
    """Placeholder environment for rollout development.
    State: embeds tool results as text via the model's embedding layer → [1, n_tokens, hidden].
    Reward: always nan (no reward signal yet).
    Will eventually live in envs/ with a real state encoder and reward function.
    """
    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reward = math.nan
        # Initialise state as a single zero embedding so rollout.py can splice it before the first update
        hidden = model.get_input_embeddings().embedding_dim
        self._state = torch.zeros(1, 1, hidden, device=model.device, dtype=model.dtype)

    @property
    def state(self) -> torch.Tensor:
        return self._state  # [1, n_tokens, hidden]

    def update(self, results: list[tuple[str, object]]) -> "DummyEnv":
        """Encode tool results as state embedding. Returns self for chaining."""
        # Represent results as readable text, tokenise, then embed
        text = "; ".join(f"{name}: {result}" for name, result in results) if results else "<no results>"
        ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            self._state = self.model.get_input_embeddings()(ids)  # [1, n_tokens, hidden]
        return self