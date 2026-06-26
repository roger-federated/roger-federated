"""roger.federated.server — the aggregation server.

Wire-compatible with the existing gradient-sharing client (`roger.federated.transport`): it seals
secure-aggregation cohorts, sums the masked uploads (masks cancel), folds η·mean(ΔW) into a
per-model cumulative global, and broadcasts that global. Run it with `python -m roger.federated.server`.
Requires the optional `[server]` extra (fastapi + uvicorn).
"""
from roger.federated.server.aggregate import Aggregator
from roger.federated.server.app import create_app

__all__ = ["Aggregator", "create_app"]
