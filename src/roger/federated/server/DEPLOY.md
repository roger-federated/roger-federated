# Running a Roger aggregation server

An aggregation server is what makes a *federation*: it seals secure-aggregation cohorts, sums the
masked client uploads (individual gradients stay hidden), and serves the cumulative global ΔW that
members fold into their base model. Anyone can run one — a federation is just its URL.

The server is **intrinsically single-instance**: secure aggregation only works when every cohort member
reaches the *same* process (one in-memory masked sum, one barrier), and concurrency is handled by
spawning rounds *inside* that one process — never by adding processes. So you want exactly **0 or 1**
instance, never more. That makes it a perfect fit for a **scale-to-zero container**: it runs the one
process while there's traffic and costs ~nothing while idle. The only durable state — the cumulative
global ΔW — lives in **S3-compatible object storage**, so it survives the container scaling to zero.

The recommended host is a managed, EU-sovereign, scale-to-zero container platform:
[**Scaleway Serverless Containers**](https://www.scaleway.com/en/serverless-containers/) (primary) or
**Koyeb** — both run the image below unchanged, scale to zero, and offer S3-compatible object storage.
A legacy always-on-VM recipe is kept at the bottom for self-hosters.

## You need
- An **object-storage bucket** (Scaleway Object Storage / Koyeb / any S3-compatible store) plus an
  access key + secret for it. The container holds these creds; **clients never touch storage** (no
  client keys, no user accounts).
- A **container registry** to push the image to (the platform's own registry is simplest).
- The platform serves HTTPS on a generated URL; that URL *is* the federation.

## 1. Build and push the image
The repo root is the build context:
```bash
docker build -f src/roger/federated/server/Dockerfile -t roger-agg .
# tag + push to your platform's registry, e.g. Scaleway:
#   docker tag roger-agg rg.fr-par.scw.cloud/<namespace>/roger-agg:latest
#   docker push        rg.fr-par.scw.cloud/<namespace>/roger-agg:latest
```
The image bundles CPU-only PyTorch + boto3 and runs `python -m roger.federated.server` (uvicorn on
`0.0.0.0:8000`).

## 2. Deploy as a scale-to-zero container
Create a Serverless Container from the pushed image with these settings:

| Setting | Value | Why |
|---|---|---|
| **max instances** | **1** | **Load-bearing.** A 2nd instance would split a cohort across processes and the secure-agg masks would never cancel. Never scale out. |
| min instances | 0 | Scale to zero when idle = ~zero cost. (Bump to 1 only while a federation is *actively* busy, to skip cold-starts during live cohorts.) |
| concurrency | high (e.g. 80) | All cohort members must share the one instance; serve them concurrently rather than spilling to a new instance. |
| request timeout | **≥ `ROGER_AGG_W` + a margin** (e.g. 60 s) | `/round/register` long-polls until the cohort seals (up to `W`, default 20 s). A shorter platform timeout would kill the barrier mid-seal. Keep `W` well under the client's 30 s call timeout. |
| memory | enough to hold one model's global ΔW + the fold (a few hundred MB–GB depending on model/LoRA target set) | The fold loads the global into RAM. |
| port | 8000 | What uvicorn binds. |

Set the environment variables (storage + the aggregation knobs from the next section):
```
ROGER_SERVER_STORAGE=s3
ROGER_S3_ENDPOINT=https://s3.fr-par.scw.cloud      # your provider's S3 endpoint
ROGER_S3_REGION=fr-par
ROGER_S3_BUCKET=roger-global
ROGER_S3_KEY=...                                    # object-storage access key
ROGER_S3_SECRET=...                                 # object-storage secret
# ROGER_S3_PREFIX=                                  # optional key prefix to share one bucket across deploys
ROGER_AGG_TRUSTED_PROXIES=<platform proxy CIDR>     # the platform terminates TLS in front of the container
```
The platform provisions TLS and fronts the container, so set `ROGER_AGG_TRUSTED_PROXIES` to its
forwarding-proxy range for the real client IP (the IP-binding check); leave it default only if you do
not rely on IP-binding.

**Cold start:** after idle, the first request pays a cold-start (pull image + rehydrate the global from
object storage); subsequent cohort members arrive warm. A live cohort holds the instance warm via the
open `/round/register` long-polls, so an in-flight round is not dropped. Losing in-memory density/round
state on scale-to-zero is harmless — idle means "bootstrap" is the correct mode anyway.

Members then add the URL to `~/.roger/config.json`:
`"federations": ["https://<your-container-url>"]`. Smoke-test:
`curl -i "https://<your-container-url>/global?model_id=x"` should return **204** before any upload.

## Configuration (environment variables)
**Storage**
| Var | Default | Meaning |
|---|---|---|
| `ROGER_SERVER_STORAGE` | `fs` | `s3` for the scale-to-zero deploy (durable global in object storage); `fs` for a local-disk/always-on VM. |
| `ROGER_S3_ENDPOINT` / `_REGION` / `_BUCKET` / `_KEY` / `_SECRET` | — | S3-compatible object store (required when `=s3`). |
| `ROGER_S3_PREFIX` | `` | Optional key prefix so several deployments can share one bucket. |
| `ROGER_SERVER_DATA` | `./server-data` | Local data dir (only used when `ROGER_SERVER_STORAGE=fs`). |
| `ROGER_SERVER_HOST` / `ROGER_SERVER_PORT` | `0.0.0.0` / `8000` | uvicorn bind. |

**Aggregation knobs**
| Var | Default | Meaning |
|---|---|---|
| `ROGER_AGG_KMIN` | `3` | Min cohort size to seal (a cohort of 1 is unmasked; at 2 each peer can subtract its own ΔW to recover the other's). Also the collusion margin: unmasking one member needs `KMIN−1` colluding peers. |
| `ROGER_AGG_KTARGET` | `5` | Seal immediately at this many registrants. Also the dropout blast radius (a no-show voids its whole cohort), so keep it modest. Defaults to the busy-mode threshold below. |
| `ROGER_AGG_W` | `20` | Registration window, seconds — must stay below the client's 30 s timeout *and* below the platform's request timeout. |
| `ROGER_AGG_U` | `20` | Seconds to wait for every sealed member to upload before voiding the round. |
| `ROGER_AGG_ETA` | `1.0` | Server learning rate: `G ← G + η·mean(ΔW)`. Lower to damp noisy rounds. |
| `ROGER_AGG_ETA_BOOT` | *(=`ETA`)* | Learning rate for a single async bootstrap upload (`G ← G + η_boot·clip(ΔW)`, k=1). Lower it to damp the noisier per-upload bootstrap gradients. |
| `ROGER_AGG_CLIP` | `1.0` | Per-client L2 budget; the cohort sum is clipped to `cohort_size · CLIP` (and a single bootstrap upload to `CLIP`). |
| `ROGER_AGG_BUSY_THRESHOLD` | *(=`KTARGET`)* | Distinct recent contributors needed to switch a model from bootstrap (async DP) to busy (secure-agg cohorts). |
| `ROGER_AGG_BUSY_WINDOW` | `180` | Rolling window, seconds, over which those distinct contributors are counted. |
| `ROGER_AGG_IPBIND` | `1` | Reject an upload whose source IP never registered (best-effort only). |
| `ROGER_AGG_MODELS` | *(any)* | Comma-separated `model_id` allowlist; empty accepts any base model. |
| `ROGER_AGG_TRUSTED_PROXIES` | `127.0.0.1/32,::1/128,10.0.0.0/8` | Proxies whose `X-Forwarded-For` is trusted for the real client IP — set to your platform's proxy range. |

## Notes & limits
- **Cold-start is handled by bootstrap mode.** While a model has fewer than `BUSY_THRESHOLD` recent
  contributors, `/status` reports `bootstrap` and clients upload a single DP-noised, *unmasked* ΔW to
  `/contribute_dp`, folded asynchronously — no cohort, no arrival coincidence, no 503. Privacy then
  rests on the client's (faux-)DP factor noise, not secure aggregation, so bootstrap is a temporary
  obfuscation regime: expect noisy, modest per-upload gradients until the federation fills up. Once
  `BUSY_THRESHOLD` distinct contributors appear within `BUSY_WINDOW`, the model flips to busy mode.
- In **busy** mode a round only aggregates when ≥`KMIN` members register and upload within the same
  ~`W`-second window; a sub-`KMIN` cohort gets a 503 and retries (safe behaviour — it would otherwise
  expose an individual ΔW).
- No dropout recovery yet: one sealed member that never uploads voids only its own cohort.
- Federations are open and unauthenticated; IP-binding is weak. Secure aggregation hides individual
  ΔW but can't filter a *well-formed* poisoned upload — only the aggregate norm-clip, small cohorts,
  and small `ETA` bound the damage. Strong per-client bounds need ZK range proofs (future work).
- **State durability:** only the cumulative global is persisted (to object storage in `s3` mode). Open
  cohorts and the bootstrap↔busy density window are in-memory and intentionally ephemeral.

## Legacy: always-on VM (self-hosted, `fs` storage)
If you'd rather run a plain VM (no object storage, fixed monthly cost), use local-disk storage and put
Caddy in front for TLS:
```bash
docker run -d --name roger-agg --restart unless-stopped \
    -p 127.0.0.1:8000:8000 -v roger-agg-data:/data \
    -e ROGER_SERVER_STORAGE=fs -e ROGER_SERVER_DATA=/data roger-agg
```
Without Docker: `pip install -e ".[server]"` then `python -m roger.federated.server`. Front it with a
reverse proxy that provisions TLS — Caddy is simplest, a minimal `/etc/caddy/Caddyfile`:
```
roger.example.org {
    reverse_proxy 127.0.0.1:8000
}
```
Open inbound 80 (cert challenge) and 443; point an `A`/`AAAA` record at the VM. `--restart
unless-stopped` plus the data volume keeps the cumulative global across reboots. The bundled `Caddyfile`
is a starting point; it is not used in the managed-container deploy above.
