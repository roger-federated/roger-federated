# federated/ — gradient sharing. Client: densify a round's LoRA update into a weight-space ΔW
# (delta.py), mask it with Bonawitz et al. secure aggregation (secure_agg.py), upload + pull the
# aggregated global per federation (transport.py), orchestrated by client.py. Server (server/): seals
# secure-aggregation cohorts, sums the masked uploads, and folds η·mean(ΔW) into a per-model
# cumulative global it broadcasts back (all-or-nothing rounds, aggregate norm-bound; Shamir dropout
# recovery + membership auth are future work, see readme + the federated-server-roadmap memory).

# Wire-protocol version this client speaks. A plain monotonic counter (NOT the pyproject marketing
# version) — bump it only when a federation-protocol change makes older clients' contributions
# incompatible. Federations advertise `min_client` (hard floor: below it we skip contributing to that
# fed, exactly like an unsupported model) and `latest_client` (advisory: below it we print an update
# notice) at /status. Derived per-fed in client.probe_federations; surfaced by cli.py at startup+quit.
CLIENT_VERSION = 1
