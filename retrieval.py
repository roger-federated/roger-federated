"""Lexical BM25 retrieval over working-directory files for auto-triggered RAG.

Public API: build_index → retrieve → format_context → mark_injected.
`score` is the pluggable scorer seam; swap it for dense/hybrid without touching
the rollout wiring.
"""
import os, re, math

_TEXT_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".json", ".yaml",
    ".yml", ".toml", ".sh", ".ps1", ".html", ".css", ".java", ".go",
    ".rs", ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".scala", ".kt",
}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", ".venv",
              ".mypy_cache", ".ruff_cache", "dist", "build", ".next"}
_BM25_K1 = 1.5
_BM25_B  = 0.75
_LOC_MAX = 40   # max lines returned by localizer
_LOC_PAD = 2    # context lines added around the trimmed match span


def tokenize_lexical(text: str) -> list[str]:
    """Lowercase + split on non-alnum; also split camelCase and snake_case identifiers.

    'readFile' → ['read', 'file'],  'read_file' → ['read', 'file'].
    Tokens shorter than 2 chars dropped (noise).
    """
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)   # camelCase split
    tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 1]


def build_index(root: str = None, max_bytes: int = 262144) -> dict:
    """Build a file-level BM25 index over text files under `root`.

    Each doc stores only {"path", "tf": dict, "dl": int} — no raw text in memory.
    Returns {"docs", "idf", "avgdl", "N", "root"}.
    IDF uses df = files-containing-term (textbook file-level definition).
    """
    if root is None:
        root = os.getcwd()
    docs = []
    df: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() not in _TEXT_EXTS:
                continue
            fpath = os.path.join(dirpath, fname)
            if os.path.getsize(fpath) > max_bytes:
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            tokens = tokenize_lexical(text)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t in tf:   # df counts files, not occurrences
                df[t] = df.get(t, 0) + 1
            docs.append({"path": os.path.relpath(fpath, root), "tf": tf, "dl": len(tokens)})

    N = len(docs)
    avgdl = sum(d["dl"] for d in docs) / max(N, 1)
    idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}
    return {"docs": docs, "idf": idf, "df": df, "avgdl": avgdl, "N": N, "root": root}


def score(query_tokens: list[str], index: dict) -> list[float]:
    """BM25 score of `query_tokens` against every document in the index.

    Pluggable seam: replace this function (same signature) for dense or hybrid retrieval.
    """
    idf, avgdl = index["idf"], index["avgdl"]
    scores = []
    for doc in index["docs"]:
        tf, dl = doc["tf"], doc["dl"]
        s = 0.0
        for t in query_tokens:
            if t not in idf:
                continue
            f = tf.get(t, 0)
            num = idf[t] * f * (_BM25_K1 + 1)
            den = f + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avgdl, 1))
            s += num / den
        scores.append(s)
    return scores


def _localize(path: str, root: str, query_terms: set[str], idf: dict,
              max_lines: int = _LOC_MAX, pad: int = _LOC_PAD):
    """Find the densest query-term span within the file.

    Slides a max_lines window over per-line IDF-weighted scores, picks the best window,
    then trims zero-weight leading/trailing lines and adds pad context lines.
    Returns (start_1indexed, end_1indexed, text) or None on read error.
    """
    fpath = os.path.join(root, path)
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return None
    if not lines:
        return None

    # Per-line relevance weight (sum of IDF for matching query terms)
    weights = [
        sum(idf.get(t, 0.0) for t in set(tokenize_lexical(line)) & query_terms)
        for line in lines
    ]
    n = len(lines)

    if n <= max_lines:
        start, end = 0, n - 1
    else:
        # Rolling-sum slide: find the max-weight window of exactly max_lines
        cum = [sum(weights[i:i + max_lines]) for i in range(n - max_lines + 1)]
        best = cum.index(max(cum))
        start, end = best, best + max_lines - 1

    # Trim zero-weight edges inside the window, then restore padding
    while start < end and weights[start] == 0:
        start += 1
    while end > start and weights[end] == 0:
        end -= 1
    start = max(0, start - pad)
    end = min(n - 1, end + pad)

    return start + 1, end + 1, "".join(lines[start:end + 1])   # 1-indexed


def retrieve(query: str, index: dict, k: int = 3,
             min_ratio: float = 0.5, specific_df_frac: float = 0.25,
             exclude: dict = None) -> list[dict]:
    """Return up to `k` localized file spans most relevant to `query`.

    Discriminative gate: only fires when the query contains ≥1 *specific* term —
    one that appears in ≤ `specific_df_frac` of corpus files (rare ⇒ informative).
    Generic queries ("run the task", "help me") carry no specific term and return [].
    After ranking, drops tail docs whose score < `min_ratio` × top score.
    Each returned doc must contain ≥1 specific term (excludes common-word-only matches).
    `exclude`: dict[relpath → set[int]] of already-shown 1-indexed line numbers;
    a span is skipped when > 50% of its lines overlap the exclusion set.
    """
    if exclude is None:
        exclude = {}
    qtokens = tokenize_lexical(query)
    if not qtokens:
        return []
    unique = set(qtokens)

    # Discriminative gate: specific = terms present in ≤ df_cap files
    df_cap = max(1, int(specific_df_frac * index["N"]))
    specific = {t for t in unique if 0 < index["df"].get(t, 0) <= df_cap}
    if not specific:
        return []   # query is all common words — no retrieval signal

    # Score all docs but keep only those containing ≥1 specific term
    scored = [
        (s, doc) for s, doc in zip(score(qtokens, index), index["docs"])
        if s > 0 and any(doc["tf"].get(t) for t in specific)
    ]
    if not scored:
        return []
    ranked = sorted(scored, key=lambda x: -x[0])
    top = ranked[0][0]

    hits = []
    for s, doc in ranked:
        if len(hits) >= k:
            break
        if s < min_ratio * top:   # relative cutoff — drop weak tail matches
            break
        result = _localize(doc["path"], index["root"], unique, index["idf"])
        if result is None:
            continue
        start, end, text = result
        shown = exclude.get(doc["path"], set())
        span = set(range(start, end + 1))
        if shown and len(span & shown) / len(span) > 0.5:
            continue
        hits.append({"path": doc["path"], "start": start, "end": end, "text": text})
    return hits


def mark_injected(injected: dict, hits: list[dict]):
    """Record shown line ranges in the dedup ledger (mutates `injected` in-place)."""
    for h in hits:
        injected.setdefault(h["path"], set()).update(range(h["start"], h["end"] + 1))


def format_context(hits: list[dict]) -> str:
    """Render retrieved hits as a context block for prompt injection.

    Returns "" when hits is empty — callers can guard with a plain truthiness check.
    """
    if not hits:
        return ""
    parts = ["[Retrieved working-directory context]"]
    for h in hits:
        parts.append(f"--- {h['path']}:{h['start']}-{h['end']} ---")
        parts.append(h["text"].rstrip())
    return "\n".join(parts)
