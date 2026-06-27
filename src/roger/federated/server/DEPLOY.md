# Running a Roger aggregation server

An aggregation server is what makes a *federation*: it seals secure-aggregation cohorts, sums the
masked client uploads (individual gradients stay hidden), and serves the cumulative global ΔW that
members fold into their base model. Anyone can run one — a federation is just its URL. It's a single
small always-on process that needs a public HTTPS endpoint and a little persistent state; it can't be
serverless because `/round/register` long-polls while a cohort fills and round state lives in memory.
One small VM is the whole footprint (any provider; pick an EU one if you care about data residency).

## You need
- A small VM (~2 vCPU / 2 GB RAM — the image bundles CPU-only PyTorch) with Docker installed.
- A domain (or subdomain) with an `A`/`AAAA` record pointing at the VM — clients require HTTPS.
- Inbound TCP **80** and **443** open (80 for the TLS cert challenge, 443 to serve).

## Run
```bash
# from a clone of this repo (the repo root is the build context); server binds loopback only
docker build -f src/roger/federated/server/Dockerfile -t roger-agg .
docker run -d --name roger-agg --restart unless-stopped \
    -p 127.0.0.1:8000:8000 -v roger-agg-data:/data roger-agg
```
Without Docker: `pip install -e ".[server]"` then `python -m roger.federated.server` (serves
`0.0.0.0:8000`; data under `$ROGER_SERVER_DATA`).

Put a reverse proxy with automatic TLS in front. Caddy is simplest — a minimal `/etc/caddy/Caddyfile`:
```
roger.example.org {
    reverse_proxy 127.0.0.1:8000
}
```
Reload Caddy; it provisions a Let's Encrypt cert once the domain resolves to the VM and 80/443 are
open. The bundled `Caddyfile` is a starting point.

Members then add the URL to `~/.roger/config.json`: `"federations": ["https://roger.example.org"]`.
Smoke-test: `curl -i https://roger.example.org/global?model_id=x` should return **204**.

## Configuration (env vars on `docker run -e …`)
| Var | Default | Meaning |
|---|---|---|
| `ROGER_AGG_KMIN` | `3` | Min cohort size to seal (a cohort of 1 is unmasked; at 2 each peer can subtract its own ΔW to recover the other's). Also the collusion margin: unmasking one member needs `KMIN−1` colluding peers. |
| `ROGER_AGG_KTARGET` | `5` | Seal immediately at this many registrants. Also the dropout blast radius (a no-show voids its whole cohort), so keep it modest. Defaults to the busy-mode threshold below. |
| `ROGER_AGG_W` | `20` | Registration window, seconds — must stay below the client's 30 s timeout. |
| `ROGER_AGG_U` | `20` | Seconds to wait for every sealed member to upload before voiding the round. |
| `ROGER_AGG_ETA` | `1.0` | Server learning rate: `G ← G + η·mean(ΔW)`. Lower to damp noisy rounds. |
| `ROGER_AGG_ETA_BOOT` | *(=`ETA`)* | Learning rate for a single async bootstrap upload (`G ← G + η_boot·clip(ΔW)`, k=1). Lower it to damp the noisier per-upload bootstrap gradients. |
| `ROGER_AGG_CLIP` | `1.0` | Per-client L2 budget; the cohort sum is clipped to `cohort_size · CLIP` (and a single bootstrap upload to `CLIP`). |
| `ROGER_AGG_BUSY_THRESHOLD` | *(=`KTARGET`)* | Distinct recent contributors needed to switch a model from bootstrap (async DP) to busy (secure-agg cohorts). |
| `ROGER_AGG_BUSY_WINDOW` | `180` | Rolling window, seconds, over which those distinct contributors are counted. Keep it a small multiple of `W`: cohorts only seal within `W`, so counting over hours would flip to busy on traffic too spread out to ever seal. |
| `ROGER_AGG_IPBIND` | `1` | Reject an upload whose source IP never registered (best-effort only). |
| `ROGER_AGG_MODELS` | *(any)* | Comma-separated `model_id` allowlist; empty accepts any base model. |
| `ROGER_AGG_TRUSTED_PROXIES` | `127.0.0.1/32,::1/128,10.0.0.0/8` | Proxies whose `X-Forwarded-For` is trusted for the real client IP — set to your proxy's address. |

Storage/bind: `ROGER_SERVER_DATA` (default `/data`, persist it), `ROGER_SERVER_HOST`/`PORT`
(default `0.0.0.0:8000`).

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
- Reboot-safe: `--restart unless-stopped` plus the data volume keeps the cumulative global; rebuild
  the image to update and the volume carries `G` across.
