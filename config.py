"""config.py — Persistent user configuration for Roger (.roger/config.json in cwd).

Loaded once at startup; CLI flags override individual keys for that run.
Users edit the JSON file directly to change defaults persistently.
"""
import json, os

def _config_path() -> str:
    return os.path.join(os.getcwd(), ".roger", "config.json")

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
    if not os.path.exists(_config_path()):
        os.makedirs(os.path.dirname(_config_path()), exist_ok=True)
        with open(_config_path(), "w") as f:
            json.dump(_DEFAULTS, f, indent=2)
        return dict(_DEFAULTS)
    with open(_config_path()) as f:
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
    os.makedirs(os.path.dirname(_config_path()), exist_ok=True)
    with open(_config_path(), "w") as f:
        json.dump(cfg, f, indent=2)


def path() -> str:
    return _config_path()
