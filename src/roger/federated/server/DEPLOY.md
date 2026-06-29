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

> **Read the [memory sizing](#memory-sizing) section.** The server now stages every upload to object
> storage and aggregates **one module at a time**, so peak RAM is ~a single weight matrix at *any* model
> size — memory is no longer the binding constraint. For large base models the constraint shifts to the
> **per-round S3 I/O** (each round moves the cohort's uploads through the bucket), so size `ROGER_AGG_U`
> and your bucket bandwidth accordingly. Also add the **`tmp/` lifecycle rule** (below) so a crashed
> round can't leave staged objects behind.

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
   blob per model each round; versioning just retains dead copies forever). Add a **lifecycle rule that
   expires objects under the `tmp/` prefix** after a few hours: the server streams each round's uploads
   to `tmp/<round_id>/` and deletes them at finalize, but a container killed mid-round (scale-to-zero,
   crash) would otherwise orphan them. The rule is the backstop GC.
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
| CPU | modest (≈1 vCPU) | The server does **no inference** — only per-module tensor sums + safetensors I/O. CPU is not the constraint. |
| memory | see [memory sizing](#memory-sizing) | Per-module aggregation keeps peak RAM to ~one weight matrix at any model size; a small tier (e.g. 2–4 GB) suffices. |
| port | 8000 | What uvicorn binds. |

**Cold start:** after idle, the first request pays a cold-start (pull image + start uvicorn); the global
is read from object storage lazily, only for the model actually requested (nothing is pre-loaded).
Subsequent cohort members arrive warm. A live cohort holds the instance warm via the open
`/round/register` long-polls, so an in-flight round is not dropped. Losing in-memory density/round state
on scale-to-zero is harmless — idle means "bootstrap" is the correct mode anyway. (Uploads already
staged to `tmp/` survive a scale-to-zero, but their round's cohort barrier does not, so the round voids
and the `tmp/` lifecycle rule reclaims them.)

## IP-binding behind a managed platform
A managed platform terminates TLS in front of the container, so the real client IP arrives via
`X-Forwarded-For`. The server only trusts that header from proxies listed in `ROGER_AGG_TRUSTED_PROXIES`
— but these platforms rarely publish a *stable* proxy CIDR to whitelist. IP-binding is best-effort
anyway, so set **`ROGER_AGG_IPBIND=0`** rather than trusting `0.0.0.0/0` (which would let any client
spoof its IP — strictly worse than turning the check off). On a VM where you control the proxy (next
section), the default trusted set covers localhost and IP-binding can stay on.

## Memory sizing
Memory used to be the load-bearing constraint; two changes removed it.

- **Narrow basis.** The client trains the federated LoRA on `q_proj`/`v_proj` only (`lora_utils.FED_TARGETS`),
  the fixed dense basis every member shares. That is ~a few percent of an `all-linear` target set, so the
  cumulative dense global and each masked upload are correspondingly small.
- **Stage + aggregate per module.** A secure-agg round must sum every member's masked vector over the full
  dense basis; summed in RAM that is `8·P_target` bytes (int64) resident for the whole collection window,
  × concurrent cohorts — which is what blew past serverless tiers. Instead the server **streams each upload
  to its own object** under `tmp/<round_id>/` and, at finalize, reads back **one module at a time**
  (range-GET), sums it (masks cancel per coordinate), folds it into the global, and streams the new global
  out via multipart. Peak RAM is ~one weight matrix plus small buffers, at **any** model size, and
  concurrent cohorts no longer multiply it (their partial state lives in the bucket, not RAM). The server
  also no longer pre-loads every model's global; it touches only the model in the request.

Consequences:
- A **small memory tier suffices** (2–4 GB is ample). **Ephemeral/scratch storage is irrelevant** — on
  Scaleway Serverless Containers it is RAM-backed tmpfs (writing N bytes to `/tmp` raises memory use by N
  and counts against the limit), so it was never a real spill target; we stage to the S3 bucket instead.
- The remaining constraint for **large** models is **per-round S3 I/O**, not RAM: each round moves the
  cohort's uploads (`~k × P_target` int64) into the bucket and reads them back to aggregate. With the q/v
  basis this is modest for small/mid models; for the very largest, give `ROGER_AGG_U` more headroom and
  expect more S3 request volume. `ROGER_AGG_MODELS` still lets you allowlist which base models a given
  deployment serves.

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
| `ROGER_AGG_ETA_BOOT` | *(=`ETA`)* | Learning rate for a single async bootstrap upload (`G ← G + η_boot·ΔW`, k=1). Lower it to damp the noisier per-upload bootstrap gradients. |
| `ROGER_AGG_CLIP` | `1.0` | Per-client L2 budget. The server **voids** a round whose aggregate `‖ΣΔW‖` exceeds `cohort_size · CLIP` (a bootstrap upload exceeding `CLIP`). Honest clients clip below this, so only a non-clipping client trips it. |
| `ROGER_AGG_BUSY_THRESHOLD` | *(=`KTARGET`)* | Distinct recent contributors needed to switch a model from bootstrap (async DP) to busy (secure-agg cohorts). |
| `ROGER_AGG_BUSY_WINDOW` | `180` | Rolling window, seconds, over which those distinct contributors are counted. |
| `ROGER_AGG_IPBIND` | `1` | Reject an upload whose source IP never registered (best-effort only). Set `0` on managed platforms without a stable proxy CIDR. |
| `ROGER_AGG_MODELS` | *(any)* | Comma-separated `model_id` allowlist; empty accepts any base model. Scope which models a deployment serves (e.g. to bound per-round S3 I/O for very large models; see memory sizing). |
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
  ΔW but can't filter a *well-formed* poisoned upload — only the aggregate norm bound (over-norm rounds
  are voided), small cohorts, and small `ETA` bound the damage. Strong per-client bounds need ZK range
  proofs (future work).
- **State durability:** only the cumulative global is durable (one blob + version per model). A round's
  in-flight uploads are staged under `tmp/<round_id>/` and deleted at finalize; open cohorts and the
  bootstrap↔busy density window are in-memory and intentionally ephemeral. The bucket `tmp/` lifecycle
  rule is the only cleanup for uploads orphaned by a mid-round crash.

## Legacy: always-on VM (self-hosted, `fs` storage)
If you'd rather run a plain VM (no object storage, fixed monthly cost), use local-disk storage and put
a TLS-terminating reverse proxy in front. The `fs` backend stages uploads under `<data>/tmp/` and cleans
them per round, just like `s3`; give the volume room for a round's in-flight uploads. Cost: paying 24/7.
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
