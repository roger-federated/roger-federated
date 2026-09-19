"""config.py — persistent user configuration for Roger (~/.roger/config.json).

Global per-user config, read by the runtime wrapper at launch. Users edit the JSON file directly.
Baseline defaults ship as package data (roger/config.json), so they are available after install.
"""
import json, os
from importlib.resources import files

from roger.paths import state_dir


def _config_path() -> str:
    return os.path.join(state_dir(), "config.json")

# Read shipped baseline once at import (package data, not the user's cwd config)
_DEFAULTS = json.loads(files("roger").joinpath("config.json").read_text(encoding="utf-8"))


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
