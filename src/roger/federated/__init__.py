# federated/ — client-side gradient sharing: densify a round's LoRA update into a weight-space ΔW
# (delta.py), mask it with Bonawitz et al. secure aggregation (secure_agg.py), upload + pull the
# aggregated global per federation (transport.py), orchestrated by client.py. The aggregation server
# is future work (see the federated_server_requirements memory).
