"""store.py — durable persistence of the cumulative global ΔW (one per model_id).

The server's only durable state is the running global `G_model = Σ_rounds η·mean(ΔW)`; rounds
themselves are ephemeral in-memory state (lost on restart, which is fine — a dropped round just
isn't aggregated). We persist `G` as the exact dense-ΔW safetensors blob the client expects to pull
(`delta.to_bytes`), plus a monotonic version integer that doubles as the `/global` cursor.

Two interchangeable backends behind the same `save_global`/`load_all` signatures, chosen by
`ROGER_SERVER_STORAGE`:
  fs  (default) — local filesystem under `datadir`; used by tests and a self-hosted always-on VM.
  s3  — any S3-compatible object store (Scaleway/Koyeb/...). This is what makes the server safe to run
        as a *scale-to-zero container*: the container's disk is ephemeral and vanishes when it scales
        to zero, so the global must live off-box. The container holds the bucket credentials, so
        clients never touch storage directly (no client keys, no per-user accounts). boto3 is imported
        lazily so the fs path — and every non-server install — never needs it.
"""
import hashlib, os, re


def _backend() -> str:
    return os.environ.get("ROGER_SERVER_STORAGE", "fs").lower()


def _stem(model_id: str) -> str:
    # model_id contains '/' (e.g. google/gemma-4-E2B-it); make a key/filename-safe, collision-resistant
    # stem (readable prefix + short hash so two ids can never alias after sanitization).
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id) + "-" + hashlib.sha1(model_id.encode()).hexdigest()[:8]


def save_global(datadir: str, model_id: str, tensors: dict, version: int) -> None:
    if _backend() == "s3":
        _s3_save(model_id, tensors, version)
    else:
        _fs_save(datadir, model_id, tensors, version)


def load_all(datadir: str) -> dict:
    """Rehydrate {model_id: (tensors, version)} at startup (and, for a scale-to-zero container, on every
    cold start). model_id is recovered from each blob's metadata (delta.to_bytes stamps it), so the
    on-disk key/filename never has to be reversed."""
    return _s3_load_all() if _backend() == "s3" else _fs_load_all(datadir)


# --- fs backend ------------------------------------------------------------------------------

def _fs_paths(datadir: str, model_id: str) -> tuple[str, str]:
    base = os.path.join(datadir, _stem(model_id))
    return base + ".safetensors", base + ".version"


def _fs_save(datadir: str, model_id: str, tensors: dict, version: int) -> None:
    from roger.federated import delta
    os.makedirs(datadir, exist_ok=True)
    st_path, v_path = _fs_paths(datadir, model_id)
    tmp = st_path + ".tmp"                        # write-then-rename so a crash never leaves a torn blob
    with open(tmp, "wb") as f:
        f.write(delta.to_bytes(tensors, model_id))
    os.replace(tmp, st_path)
    with open(v_path, "w") as f:
        f.write(str(version))


def _fs_load_all(datadir: str) -> dict:
    from roger.federated import delta
    out: dict = {}
    if not os.path.isdir(datadir):
        return out
    for fn in os.listdir(datadir):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(datadir, fn)
        with open(path, "rb") as f:
            tensors, meta = delta.from_bytes(f.read())
        mid = meta.get("model_id")
        if not mid:
            continue
        v_path = path[: -len(".safetensors")] + ".version"
        version = int(open(v_path).read()) if os.path.exists(v_path) else 1
        out[mid] = (tensors, version)
    return out


# --- s3 backend ------------------------------------------------------------------------------
# An optional ROGER_S3_PREFIX lets several deployments share one bucket; datadir is unused here (it is
# a local-FS concept). Pagination is ignored on purpose: a federation serves a handful of base models,
# far under the 1000-key list cap.

def _s3_conf() -> tuple:
    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["ROGER_S3_ENDPOINT"],
        region_name=os.environ.get("ROGER_S3_REGION", "fr-par"),
        aws_access_key_id=os.environ["ROGER_S3_KEY"],
        aws_secret_access_key=os.environ["ROGER_S3_SECRET"],
    )
    return client, os.environ["ROGER_S3_BUCKET"], os.environ.get("ROGER_S3_PREFIX", "")


def _s3_save(model_id: str, tensors: dict, version: int) -> None:
    from roger.federated import delta
    client, bucket, prefix = _s3_conf()
    stem = prefix + _stem(model_id)
    # put_object is atomic per key; write the blob before the version so a crash between them leaves the
    # version trailing (re-served as the older/served global), never a torn blob — same ordering as fs.
    client.put_object(Bucket=bucket, Key=stem + ".safetensors", Body=delta.to_bytes(tensors, model_id))
    client.put_object(Bucket=bucket, Key=stem + ".version", Body=str(version).encode())


def _s3_load_all() -> dict:
    from roger.federated import delta
    client, bucket, prefix = _s3_conf()
    out: dict = {}
    listing = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in listing.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".safetensors"):
            continue
        tensors, meta = delta.from_bytes(client.get_object(Bucket=bucket, Key=key)["Body"].read())
        mid = meta.get("model_id")
        if not mid:
            continue
        v_key = key[: -len(".safetensors")] + ".version"
        try:
            version = int(client.get_object(Bucket=bucket, Key=v_key)["Body"].read())
        except Exception:
            version = 1                          # missing/unreadable version sidecar -> treat as first
        out[mid] = (tensors, version)
    return out
