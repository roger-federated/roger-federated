# federated/ — gradient sharing, client side. A training round's LoRA-factor update is masked with
# Bonawitz et al. secure aggregation (secure_agg.py) and uploaded to one federation (client.py) over
# the fail-soft HTTP surface in transport.py; delta.py holds the factor/serialization contract the
# server mirrors. The pull half is torch-free and lives in runtime/adapter.py, which turns the
# federation's global into the adapter the runtime loads.

# Wire-protocol version this client speaks. A plain monotonic counter (NOT the pyproject marketing
# version) — bump it when a federation-protocol change makes older clients' contributions incompatible,
# or when a build must be force-adopted for compliance reasons the server needs to enforce (e.g. the
# privacy notice in runtime/notice.privacy_notice: bumping here + raising the deployed server's
# ROGER_MIN_CLIENT retroactively blocks pre-notice clients from contributing, same mechanism as an
# actual protocol break). Federations advertise `min_client` (hard floor: below it we skip contributing
# to that fed, exactly like an unsupported model) and `latest_client` (advisory: below it we print an
# update notice) at /status; both are surfaced by runtime/notice.py once the runtime is up, and
# min_client is re-checked by runtime/train.py when it picks the federation to train for.
# (Not every federation feature forces a bump: the /round/register token that binds a cohort
# registration to its later upload degrades gracefully — an old client omits it and simply gets
# rejected at /contribute, same as any other fail-soft skip — so it did not itself require one; the
# bump to 2 here is the privacy-notice compliance adoption described above.)
CLIENT_VERSION = 2

# How to bring this client up to date when a federation's min_client/latest_client says so.
UPDATE_CMD = "git fetch origin && git reset --hard origin/main && uv tool install . --reinstall   (in your roger-federated clone)"
