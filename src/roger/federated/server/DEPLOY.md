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

Any managed scale-to-zero container platform works (Scaleway Serverless Containers, Koyeb, ...). The
instructions below are deliberately platform-agnostic; map them onto your provider's console or CLI.

> **Read the [memory sizing](#memory-sizing) section before you pick which models to serve.** With the
> client's `all-linear` LoRA, the dense global is roughly the model's whole non-embedding weight set,
> and the server holds every model's global in RAM. Large base models do **not** fit a scale-to-zero
> container today. This is the one thing that will bite you if skipped.

## What you need
- An **S3-compatible object-storage bucket** plus an access key + secret. The container holds these
  creds; **clients never touch storage** (no client keys, no user accounts). Some scale-to-zero
  platforms have no managed object storage of their own — that's fine, point the server at a bucket
  from any S3-compatible provider; only the endpoint/creds differ.
- A **container registry** the platform can pull from (the platform's own is usually simplest).
- The platform provisions HTTPS on a generated URL; that URL *is* the federation.

## Steps
1. **Build and push the image.** Build context is the repo root, then tag and push to your registry
   (example values are a real Scaleway registry namespace; substitute your own):
   ```bash
   docker build -f src/roger/federated/server/Dockerfile -t roger-agg .
   docker tag  roger-agg rg.nl-ams.scw.cloud/roger-containers/roger-agg:latest
   docker push rg.nl-ams.scw.cloud/roger-containers/roger-agg:latest
   ```
   The image bundles CPU-only PyTorch + boto3 and runs `python -m roger.federated.server` (uvicorn on
   `0.0.0.0:8000`).
2. **Create the bucket** and an access key/secret for it. Keep it **private** (clients reach the global
   only through the server, never the bucket) and leave **versioning off** (the server overwrites one
   blob per model each round; versioning just retains dead copies forever).
3. **Deploy the container** from the image with the [settings](#container-settings) and
   [environment](#environment-variables) below.
4. **Smoke-test:** `curl -i "https://<your-url>/global?model_id=x"` should return **204** before any
   upload. Members then add the URL to `~/.roger/config.json`: `"federations": ["https://<your-url>"]`.

## Container settings
| Setting | Value | Why |
|---|---|---|
| **max instances** | **1** | **Load-bearing.** A 2nd instance would split a cohort across processes and the secure-agg masks would never cancel. Never scale out. |
| min instances | 0 | Scale to zero when idle = ~zero cost. Bump to 1 only while a federation is *actively* busy, to skip cold-starts during live cohorts. |
| concurrency | high (e.g. 80) | All cohort members must share the one instance; serve them concurrently rather than spilling to a new instance. Keep it above your largest cohort. |
| request timeout | **≥ `ROGER_AGG_W` + margin** (e.g. 60 s) | `/round/register` long-polls until the cohort seals (up to `W`, default 20 s). A shorter platform timeout would kill the barrier mid-seal. Keep `W` well under the client's 30 s call timeout. |
| CPU | modest (≈1 vCPU) | The server does **no inference** — only tensor sums + safetensors I/O, infrequently. CPU is not the constraint; memory is. |
| memory | see [memory sizing](#memory-sizing) | The global lives in RAM; this is the binding constraint and depends on the model + LoRA target set. |
| port | 8000 | What uvicorn binds. |

**Cold start:** after idle, the first request pays a cold-start (pull image + rehydrate the global from
object storage); subsequent cohort members arrive warm. A live cohort holds the instance warm via the
open `/round/register` long-polls, so an in-flight round is not dropped. Losing in-memory density/round
state on scale-to-zero is harmless — idle means "bootstrap" is the correct mode anyway.

## IP-binding behind a managed platform
A managed platform terminates TLS in front of the container, so the real client IP arrives via
`X-Forwarded-For`. The server only trusts that header from proxies listed in `ROGER_AGG_TRUSTED_PROXIES`
— but these platforms rarely publish a *stable* proxy CIDR to whitelist. IP-binding is best-effort
anyway, so set **`ROGER_AGG_IPBIND=0`** rather than trusting `0.0.0.0/0` (which would let any client
spoof its IP — strictly worse than turning the check off). On a VM where you control the proxy (next
section), the default trusted set covers localhost and IP-binding can stay on.

## Memory sizing
This is the load-bearing constraint on a scale-to-zero deploy, and it is **larger than you'd expect**.

- The cumulative global is stored **dense in weight space** (`delta.densify` reconstructs `scaling·(B@A)`
  to each module's full shape). Dense is intrinsic to secure aggregation — masked deltas can only be
  summed in a common dense basis; you can't add LoRA factors across clients.
- The client trains LoRA with `target_modules="all-linear"`, so the dense global ≈ the model's **entire
  non-embedding linear weight set**, in bf16. Rough order: ~6–8 GB for a ~5 B model, ~20 GB for a 12 B,
  tens of GB for 26–31 B models.
- The server holds **every served model's global resident in RAM** (loaded at startup and on each cold
  start, summed with the cohort's uploads in RAM). Only models that have actually received a
  contribution consume memory, but it is additive across them.

Consequences: **ephemeral/scratch storage often does not help** (on some platforms it is RAM-backed
tmpfs, counting against the same budget, so you cannot spill the global to disk), and **large base
models do not fit a scale-to-zero container** — set memory to the platform's max tier and use
`ROGER_AGG_MODELS` to allowlist only small base model(s). Large models need a different host (a big
always-on box, forfeiting the scale-to-zero win) until per-module streaming and/or a narrower LoRA
target set shrink the dense global.

## Environment variables
A filled-in `s3` example (real Scaleway region/bucket; supply your own key/secret as platform secrets):
```
ROGER_SERVER_STORAGE=s3
ROGER_S3_ENDPOINT=https://s3.nl-ams.scw.cloud   # region endpoint, no bucket prefix
ROGER_S3_REGION=nl-ams
ROGER_S3_BUCKET=roger-agg
ROGER_S3_KEY=<access key>                        # store as a secret
ROGER_S3_SECRET=<secret key>                     # store as a secret
ROGER_AGG_IPBIND=0                               # managed platform: no stable proxy CIDR
```

**Storage**
| Var | Default | Meaning |
|---|---|---|
| `ROGER_SERVER_STORAGE` | `fs` | `s3` for the scale-to-zero deploy (durable global in object storage); `fs` for a local-disk/always-on VM. |
| `ROGER_S3_ENDPOINT` / `_REGION` / `_BUCKET` / `_KEY` / `_SECRET` | — | S3-compatible object store (required when `=s3`). `ENDPOINT` is the *region* endpoint with no bucket prefix — boto3 adds the bucket from `_BUCKET`. Store `_KEY`/`_SECRET` as platform secrets, not plain env vars. |
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
| `ROGER_AGG_IPBIND` | `1` | Reject an upload whose source IP never registered (best-effort only). Set `0` on managed platforms without a stable proxy CIDR. |
| `ROGER_AGG_MODELS` | *(any)* | Comma-separated `model_id` allowlist; empty accepts any base model. Use it to keep oversized models off a scale-to-zero deploy (see memory sizing). |
| `ROGER_AGG_TRUSTED_PROXIES` | `127.0.0.1/32,::1/128,10.0.0.0/8` | Proxies whose `X-Forwarded-For` is trusted for the real client IP. On managed platforms prefer `ROGER_AGG_IPBIND=0` over widening this. |

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
a TLS-terminating reverse proxy in front. A VM also sidesteps the [memory limit](#memory-sizing) above
— give it enough RAM for the globals you serve — at the cost of paying for it 24/7.
```bash
docker run -d --name roger-agg --restart unless-stopped \
    -p 127.0.0.1:8000:8000 -v roger-agg-data:/data \
    -e ROGER_SERVER_STORAGE=fs -e ROGER_SERVER_DATA=/data roger-agg
```
Without Docker: `pip install -e ".[server]"` then `python -m roger.federated.server`. Front it with a
reverse proxy that provisions TLS — Caddy is simplest. Point an `A`/`AAAA` record at the VM, open
inbound 80 (cert challenge) and 443, and use a minimal `/etc/caddy/Caddyfile`:
```
fed.example.org {
    reverse_proxy 127.0.0.1:8000
}
```
`--restart unless-stopped` plus the data volume keeps the cumulative global across reboots.
