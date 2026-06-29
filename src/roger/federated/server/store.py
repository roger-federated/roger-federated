"""store.py — durable global persistence + per-round upload staging (fs | s3, via ROGER_SERVER_STORAGE).

Durable state is the global `G = Σ η·mean(ΔW)`, stored as the dense-ΔW safetensors blob the client pulls
(`delta.to_bytes` layout) + a version int (the /global cursor). s3 is what makes scale-to-zero safe (the
container disk is RAM-backed tmpfs and vanishes); the container holds the creds, clients never touch it.

Uploads are STAGED here one temp object each (`stage_*`) and folded one module at a time
(`open_global_reader` + streamed multipart `GlobalWriter`) so RAM is ~one matrix, not the whole model —
S3 has no in-place partial write, so the global is rebuilt streamed. Both paths (cohort + the single-
upload bootstrap `dp_fold`) stream this way. boto3 imported lazily.
"""
import hashlib, json, os, re, shutil, struct

import torch

# safetensors dtype codes <-> torch. The server only ever serializes F32 globals and reads I64 uploads.
_DTYPE_CODE = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.int64: "I64"}
_CODE_DTYPE = {v: k for k, v in _DTYPE_CODE.items()}
_DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}
_MIN_PART = 5 * 1024 * 1024            # S3 multipart minimum part size (last part may be smaller)


def _backend() -> str:
    return os.environ.get("ROGER_SERVER_STORAGE", "fs").lower()


def _stem(model_id: str) -> str:
    # model_id contains '/' (e.g. google/gemma-4-E2B-it); make a key/filename-safe, collision-resistant
    # stem (readable prefix + short hash so two ids can never alias after sanitization).
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id) + "-" + hashlib.sha1(model_id.encode()).hexdigest()[:8]


# --- safetensors header (de)serialization ----------------------------------------------------
# Hand-rolled so we can build/parse blobs without materializing whole tensors. Layout: u64 LE header
# length, JSON header {name:{dtype,shape,data_offsets:[b,e]}, "__metadata__"}, then the data segment.

def _build_header(specs: list, metadata: dict) -> bytes:
    """specs=[(key,code,shape)] in write order -> 8-byte len + padded JSON header (data follows)."""
    header, off = {}, 0
    for key, code, shape in specs:
        n = _DTYPE_SIZE[code]
        for d in shape:
            n *= d
        header[key] = {"dtype": code, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = metadata
    js = json.dumps(header, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-(8 + len(js))) % 8)        # pad so the data segment is 8-byte aligned
    return struct.pack("<Q", len(js)) + js


def _parse_header(prefix: bytes) -> tuple:
    """(tensors, metadata, data_start) from a buffer that contains at least the full header. tensors =
    {key: (code, shape_tuple, begin, end)} with offsets relative to data_start."""
    n = struct.unpack("<Q", prefix[:8])[0]
    h = json.loads(prefix[8 : 8 + n])
    meta = h.pop("__metadata__", {})
    tensors = {k: (v["dtype"], tuple(v["shape"]), v["data_offsets"][0], v["data_offsets"][1])
               for k, v in h.items()}
    return tensors, meta, 8 + n


def _slice_to_tensor(buf: bytes, code: str, shape: tuple) -> torch.Tensor:
    # bytearray(buf) makes the buffer writable so torch.frombuffer doesn't warn/share read-only memory.
    t = torch.frombuffer(bytearray(buf), dtype=_CODE_DTYPE[code])
    return t.reshape(shape) if shape else t


# --- s3 helpers -------------------------------------------------------------------------------
# An optional ROGER_S3_PREFIX lets several deployments share one bucket; datadir is unused on s3.

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


def _s3_range(client, bucket, key, start: int, length: int) -> bytes:
    return client.get_object(Bucket=bucket, Key=key,
                             Range=f"bytes={start}-{start + length - 1}")["Body"].read()


class _S3MultipartWriter:
    """Stream bytes into one S3 object via multipart upload, buffering into >=5 MiB parts so the whole
    object is never resident. complete_multipart_upload is the single atomic commit point."""
    def __init__(self, client, bucket, key):
        self.client, self.bucket, self.key = client, bucket, key
        self.upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
        self.parts, self.buf = [], bytearray()

    def write(self, b: bytes) -> None:
        self.buf += b
        if len(self.buf) >= _MIN_PART:
            self._flush()

    def _flush(self) -> None:
        if not self.buf and self.parts:        # nothing new and we already have >=1 part
            return
        n = len(self.parts) + 1
        r = self.client.upload_part(Bucket=self.bucket, Key=self.key, PartNumber=n,
                                    UploadId=self.upload_id, Body=bytes(self.buf))
        self.parts.append({"ETag": r["ETag"], "PartNumber": n})
        self.buf = bytearray()

    def commit(self) -> None:
        self._flush()                          # final (possibly <5 MiB) part
        self.client.complete_multipart_upload(Bucket=self.bucket, Key=self.key, UploadId=self.upload_id,
                                               MultipartUpload={"Parts": self.parts})

    def abort(self) -> None:
        try:
            self.client.abort_multipart_upload(Bucket=self.bucket, Key=self.key, UploadId=self.upload_id)
        except Exception:
            pass


# --- global version + per-module access ------------------------------------------------------

def load_version(datadir: str, model_id: str) -> int:
    """Current version (the /global cursor) without loading the blob; 0 if no global exists yet."""
    if _backend() == "s3":
        client, bucket, prefix = _s3_conf()
        return _s3_version(client, bucket, prefix + _stem(model_id))
    _, v_path = _fs_paths(datadir, model_id)
    return int(open(v_path).read()) if os.path.exists(v_path) else 0


def _s3_version(client, bucket, stem) -> int:
    try:
        return int(client.get_object(Bucket=bucket, Key=stem + ".version")["Body"].read())
    except Exception:
        return 0


def _fs_paths(datadir: str, model_id: str) -> tuple:
    base = os.path.join(datadir, _stem(model_id))
    return base + ".safetensors", base + ".version"


# --- broadcast: stream the stored global blob to /global -------------------------------------

def open_global_stream(datadir: str, model_id: str):
    """(chunk_iter, version) streaming the stored global's exact bytes, or None if absent. The client's
    from_bytes reads the stream identically; the server never holds the whole blob to serve it."""
    if _backend() == "s3":
        client, bucket, prefix = _s3_conf()
        stem = prefix + _stem(model_id)
        try:
            body = client.get_object(Bucket=bucket, Key=stem + ".safetensors")["Body"]
        except Exception:
            return None
        return body.iter_chunks(1 << 20), _s3_version(client, bucket, stem)
    st_path, v_path = _fs_paths(datadir, model_id)
    if not os.path.exists(st_path):
        return None
    def gen():
        with open(st_path, "rb") as f:
            while chunk := f.read(1 << 20):
                yield chunk
    version = int(open(v_path).read()) if os.path.exists(v_path) else 1
    return gen(), version


# --- per-round upload staging ----------------------------------------------------------------
# Each member's masked upload is streamed to its own temp object under tmp/<round_id>/<peer>; at
# finalize the cohort's module-m slices are range-read and summed (masks cancel). Voided/committed
# rounds are cleaned; an S3 lifecycle rule on the tmp/ prefix GCs orphans from a crashed server.

def _tmp_key(prefix: str, round_id: str, peer: str) -> str:
    return f"{prefix}tmp/{round_id}/{peer}"


def stage_writer(datadir: str, round_id: str, peer: str):
    """A streamed write sink for one member's upload (.write(chunk)/.close())."""
    if _backend() == "s3":
        client, bucket, prefix = _s3_conf()
        return _S3MultipartWriter(client, bucket, _tmp_key(prefix, round_id, peer))
    path = os.path.join(datadir, "tmp", round_id, peer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return _FsWriter(open(path, "wb"))


class _FsWriter:
    def __init__(self, fh):
        self.fh = fh
    def write(self, b: bytes) -> None:
        self.fh.write(b)
    def commit(self) -> None:
        self.fh.close()
    def abort(self) -> None:
        self.fh.close()


def stage_header(datadir: str, round_id: str, peer: str) -> tuple:
    """(data_start, flat_length) of a staged upload — its single "masked" int64 tensor."""
    head = _stage_read(datadir, round_id, peer, 0, 1 << 16)   # header is tiny (one tensor + meta)
    tensors, _, data_start = _parse_header(head)
    code, shape, b, e = tensors["masked"]
    return data_start, (e - b) // _DTYPE_SIZE[code]


def stage_read_slice(datadir: str, round_id: str, peer: str, start: int, length: int) -> bytes:
    return _stage_read(datadir, round_id, peer, start, length)


def _stage_read(datadir: str, round_id: str, peer: str, start: int, length: int) -> bytes:
    if _backend() == "s3":
        client, bucket, prefix = _s3_conf()
        return _s3_range(client, bucket, _tmp_key(prefix, round_id, peer), start, length)
    with open(os.path.join(datadir, "tmp", round_id, peer), "rb") as f:
        f.seek(start)
        return f.read(length)


def stage_cleanup(datadir: str, round_id: str) -> None:
    if _backend() == "s3":
        client, bucket, prefix = _s3_conf()
        pre = f"{prefix}tmp/{round_id}/"
        objs = client.list_objects_v2(Bucket=bucket, Prefix=pre).get("Contents", [])
        if objs:
            client.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": o["Key"]} for o in objs]})
    else:
        shutil.rmtree(os.path.join(datadir, "tmp", round_id), ignore_errors=True)


# --- per-module global read (old G) + streamed write (new G) ---------------------------------

def _fs_range(path: str, start: int, length: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(length)


def _reader_at(get_range):
    """Lazy per-module reader over a safetensors blob addressed by get_range(start,length)->bytes, or
    None if the blob is absent (the first read raises). Used for both the global and a staged DP blob —
    same dense layout. `.keys` is {key:(code,shape)}; `.read(key)` range-reads just that module."""
    try:
        n = struct.unpack("<Q", get_range(0, 8))[0]
        head = get_range(0, 8 + n)
    except Exception:
        return None
    tensors, _, data_start = _parse_header(head)
    def rd(key):
        code, shape, b, e = tensors[key]
        return _slice_to_tensor(get_range(data_start + b, e - b), code, shape)
    return _Reader(tensors, rd)


def open_global_reader(datadir: str, model_id: str):
    if _backend() == "s3":
        client, bucket, prefix = _s3_conf()
        key = prefix + _stem(model_id) + ".safetensors"
        return _reader_at(lambda s, l: _s3_range(client, bucket, key, s, l))
    st_path, _ = _fs_paths(datadir, model_id)
    return _reader_at(lambda s, l: _fs_range(st_path, s, l)) if os.path.exists(st_path) else None


def open_staged_reader(datadir: str, round_id: str, slot: str):
    """Per-module reader over a staged DP upload (a dense-ΔW blob, same format as the global)."""
    if _backend() == "s3":
        client, bucket, prefix = _s3_conf()
        key = _tmp_key(prefix, round_id, slot)
        return _reader_at(lambda s, l: _s3_range(client, bucket, key, s, l))
    path = os.path.join(datadir, "tmp", round_id, slot)
    return _reader_at(lambda s, l: _fs_range(path, s, l)) if os.path.exists(path) else None


class _Reader:
    def __init__(self, tensors: dict, rd):
        self.keys = {k: (code, shape) for k, (code, shape, _, _) in tensors.items()}
        self._rd = rd
    def read(self, key: str) -> torch.Tensor:
        return self._rd(key)


def global_writer(datadir: str, model_id: str, specs: list):
    """Open a streamed builder for the new global. specs = [(key, code, shape), ...] in write order;
    `.write(bytes)` appends each module's data in that order; `.commit(version)` atomically publishes
    the blob + version. The whole global is never resident."""
    metadata = {"model_id": model_id,
                "compat": _compat(specs)}
    header = _build_header(specs, metadata)
    if _backend() == "s3":
        client, bucket, prefix = _s3_conf()
        stem = prefix + _stem(model_id)
        w = _S3MultipartWriter(client, bucket, stem + ".safetensors")
        w.write(header)
        def publish(v):                         # complete the multipart, then write the version sidecar
            w.commit()
            client.put_object(Bucket=bucket, Key=stem + ".version", Body=str(v).encode())
        return _GlobalWriter(w, publish)
    os.makedirs(datadir, exist_ok=True)
    st_path, v_path = _fs_paths(datadir, model_id)
    tmp = st_path + ".tmp"
    fh = open(tmp, "wb")
    fh.write(header)
    def publish(v):                             # close, then rename-into-place + version (atomic-ish)
        fh.close()
        os.replace(tmp, st_path)
        with open(v_path, "w") as f:
            f.write(str(v))
    return _GlobalWriter(_FsWriter(fh), publish)


def _compat(specs: list) -> str:
    from roger.federated import delta
    return delta.compat_from_shapes({key: tuple(shape) for key, _, shape in specs})


class _GlobalWriter:
    """Streamed builder for the new global; `publish(version)` does the backend's atomic commit."""
    def __init__(self, sink, publish):
        self.sink, self._publish = sink, publish
    def write(self, b: bytes) -> None:
        self.sink.write(b)
    def commit(self, version: int) -> None:
        self._publish(version)
    def abort(self) -> None:
        self.sink.abort()
