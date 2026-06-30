"""config.py — Persistent user configuration for Roger (~/.roger/config.json).

Global per-user config (not per-project); CLI flags override individual keys for that run.
Users edit the JSON file directly to change defaults persistently. (Per-run artifacts —
runs/backups/memory/scratch — live in the per-project .roger/ instead.)

Baseline defaults ship as package data (apps/config.json) — like tools/command_policy.txt —
so they're available after install; `verbose` controls whether thinking blocks show expanded.
"""
import json, os
from importlib.resources import files
from roger.agency.path_utils import state_dir

def _config_path() -> str:
    return os.path.join(state_dir(), "config.json")

# Read shipped baseline once at import (package data, not the user's cwd config)
_DEFAULTS = json.loads(files("roger.apps").joinpath("config.json").read_text(encoding="utf-8"))


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


def default_federations() -> list[str]:
    """The federation server URLs that ship as the baseline default (apps/config.json).
    Used to warn when a user has dropped the default server and so won't pull its updates."""
    return list(_DEFAULTS.get("federations", []))
