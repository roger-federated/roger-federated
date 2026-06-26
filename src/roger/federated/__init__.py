# federated/ — gradient sharing. Client: densify a round's LoRA update into a weight-space ΔW
# (delta.py), mask it with Bonawitz et al. secure aggregation (secure_agg.py), upload + pull the
# aggregated global per federation (transport.py), orchestrated by client.py. Server (server/): seals
# secure-aggregation cohorts, sums the masked uploads, and folds η·mean(ΔW) into a per-model
# cumulative global it broadcasts back (all-or-nothing rounds, aggregate norm-bound; Shamir dropout
# recovery + membership auth are future work, see readme + the federated-server-roadmap memory).
