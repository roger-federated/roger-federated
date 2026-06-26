"""store.py — on-disk persistence of the cumulative global ΔW (one per model_id).

The server's only durable state is the running global `G_model = Σ_rounds η·mean(ΔW)`; rounds
themselves are ephemeral in-memory state (lost on restart, which is fine — a dropped round just
isn't aggregated). We persist `G` as the exact dense-ΔW safetensors blob the client expects to pull
(`delta.to_bytes`), plus a monotonic version integer that doubles as the `/global` cursor.
"""
import hashlib, os, re

from roger.federated import delta


def _paths(datadir: str, model_id: str) -> tuple[str, str]:
    # model_id contains '/' (e.g. google/gemma-4-E2B-it); make a filename-safe, collision-resistant
    # stem (readable prefix + short hash so two ids can never alias).
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model_id) + "-" + hashlib.sha1(model_id.encode()).hexdigest()[:8]
    base = os.path.join(datadir, safe)
    return base + ".safetensors", base + ".version"


def save_global(datadir: str, model_id: str, tensors: dict, version: int) -> None:
    os.makedirs(datadir, exist_ok=True)
    st_path, v_path = _paths(datadir, model_id)
    tmp = st_path + ".tmp"                       # write-then-rename so a crash never leaves a torn blob
    with open(tmp, "wb") as f:
        f.write(delta.to_bytes(tensors, model_id))
    os.replace(tmp, st_path)
    with open(v_path, "w") as f:
        f.write(str(version))


def load_all(datadir: str) -> dict:
    """Rehydrate {model_id: (tensors, version)} at startup. model_id is recovered from each blob's
    metadata (delta.to_bytes stamps it), so the on-disk filename never has to be reversed."""
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
