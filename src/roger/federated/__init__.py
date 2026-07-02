# federated/ — gradient sharing. Client: densify a round's LoRA update into a weight-space ΔW
# (delta.py), mask it with Bonawitz et al. secure aggregation (secure_agg.py), upload + pull the
# aggregated global per federation (transport.py), orchestrated by client.py. Server (server/): seals
# secure-aggregation cohorts, sums the masked uploads, and folds η·mean(ΔW) into a per-model
# cumulative global it broadcasts back (all-or-nothing rounds, aggregate norm-bound, token-based
# membership auth; Shamir dropout recovery is future work, see readme + the federated-server-roadmap
# memory).

# Wire-protocol version this client speaks. A plain monotonic counter (NOT the pyproject marketing
# version) — bump it when a federation-protocol change makes older clients' contributions incompatible,
# or when a build must be force-adopted for compliance reasons the server needs to enforce (e.g. the
# privacy notice added in cli._ensure_privacy_notice: bumping here + raising the deployed server's
# ROGER_MIN_CLIENT retroactively blocks pre-notice clients from contributing, same mechanism as an
# actual protocol break). Federations advertise `min_client` (hard floor: below it we skip contributing
# to that fed, exactly like an unsupported model) and `latest_client` (advisory: below it we print an
# update notice) at /status. Derived per-fed in client.probe_federations; surfaced by cli.py at startup+quit.
# (Not every federation feature forces a bump: the /round/register token that binds a cohort
# registration to its later upload degrades gracefully — an old client omits it and simply gets
# rejected at /contribute, same as any other fail-soft skip — so it did not itself require one; the
# bump to 2 here is the privacy-notice compliance adoption described above.)
CLIENT_VERSION = 2
