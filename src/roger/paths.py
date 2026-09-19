"""paths.py — where roger keeps its state.

Everything roger writes lives under one global per-user directory, never in the project: the config,
the runtime wrapper's captured chats (`messages/<runtime>/`), and the per-federation sync state,
global blobs and built LoRA adapters (`federated/`).
"""
import os


def state_dir() -> str:
    """Global per-user Roger state dir (~/.roger). Callers create subdirs as needed."""
    return os.path.expanduser(os.path.join("~", ".roger"))
