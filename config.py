"""config.py — Persistent user configuration for Roger (~/.roger/config.json).

Loaded once at startup; CLI flags override individual keys for that run.
Users edit the JSON file directly to change defaults persistently.
"""
import json, os

_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".roger", "config.json")

_DEFAULTS = {
    "model_id":      "google/gemma-4-E2B-it",
    "max_steps":     10,
    "max_new_tokens": 4096,
    "enable_rag":    True,
    "rag_k":         3,
    "enable_skills": True,
    "verbose":       False,   # show thinking blocks expanded
}


def load() -> dict:
    """Load config from disk; create with defaults on first run."""
    if not os.path.exists(_CONFIG_PATH):
        os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
        with open(_CONFIG_PATH, "w") as f:
            json.dump(_DEFAULTS, f, indent=2)
        return dict(_DEFAULTS)
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)
    # Fill any missing keys introduced in newer versions
    changed = False
    for k, v in _DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v; changed = True
    if changed:
        save(cfg)
    return cfg


def save(cfg: dict) -> None:
    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def path() -> str:
    return _CONFIG_PATH
