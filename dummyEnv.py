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
        self._terminal = False  # set by update() when no tool call was emitted

    def get_state(self) -> "torch.Tensor | str":
        """Observe the current state of the environment.
        Call this to obtain an up-to-date snapshot of the world after executing an action —
        e.g. the contents of a terminal, the appearance of a browser, or the output of a process.
        Use it whenever you need situational awareness before deciding on your next action."""
        if self._terminal:  # episode ended; rollout loop breaks on this sentinel
            return "terminal"
        if self._state_cache is None:
            ids = self.tokenizer.encode(
                self._state_text, return_tensors="pt", add_special_tokens=False
            ).to(self.model.device)
            with torch.no_grad():
                self._state_cache = self.model.get_input_embeddings()(ids)  # [1, n_tokens, hidden]
        return self._state_cache

    def update(self, call: dict | None, result) -> None:
        """Record the outcome of a tool call; invalidates the state cache.
        A None call (no tool call emitted, result 'terminate') ends the episode.
        Summarises as '<name>(<k>=<v>, ...) → <result>' so get_state() can embed it.
        """
        if call is None:
            self._terminal = True
            return
        args = ", ".join(f"{k}={v!r}" for k, v in call.get("arguments", {}).items())
        self._state_text = f"{call['name']}({args}) → {result}"
        self._state_cache = None

    def get_reward(self) -> float:
        """Return the current reward signal. Not implemented in this placeholder."""
        return math.nan
