import torch, math
import transformers

class DummyEnv:
    """Placeholder environment for rollout development.
    State is the embedding of a text summary of the last tool call and its result.
    Cache is invalidated on each update() and recomputed lazily by get_state().
    Will eventually live in envs/ with a real state encoder and reward function.
    """
    _NO_STATE_TEXT = "No state information available right now."

    def __init__(self, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._state_text = self._NO_STATE_TEXT
        self._state_cache: torch.Tensor | None = None  # invalidated by update()

    def get_state(self) -> torch.Tensor:
        """Observe the current state of the environment.
        Call this to obtain an up-to-date snapshot of the world after executing an action —
        e.g. the contents of a terminal, the appearance of a browser, or the output of a process.
        Use it whenever you need situational awareness before deciding on your next action."""
        if self._state_cache is None:
            ids = self.tokenizer.encode(
                self._state_text, return_tensors="pt", add_special_tokens=False
            ).to(self.model.device)
            with torch.no_grad():
                self._state_cache = self.model.get_input_embeddings()(ids)  # [1, n_tokens, hidden]
        return self._state_cache

    def update(self, call: dict, result) -> None:
        """Record the outcome of a tool call; invalidates the state cache.
        Summarises as '<name>(<k>=<v>, ...) → <result>' so get_state() can embed it.
        """
        args = ", ".join(f"{k}={v!r}" for k, v in call.get("arguments", {}).items())
        self._state_text = f"{call['name']}({args}) → {result}"
        self._state_cache = None

    def get_reward(self) -> float:
        """Return the current reward signal. Not implemented in this placeholder."""
        return math.nan
