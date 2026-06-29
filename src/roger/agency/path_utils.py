"""path_utils.py — @path reference expansion + global state-dir location.

Public API:
  expand_at_references(text, root) — expand @path tokens; append referenced blocks
  state_dir() — global per-user Roger state directory (~/.roger)
"""
import os, re


def state_dir() -> str:
    """Global per-user Roger state dir (~/.roger). Callers create subdirs as needed."""
    return os.path.expanduser(os.path.join("~", ".roger"))


_MAX_BYTES = 262144                     # mirrors retrieval.build_index max_bytes
_MAX_DEPTH = 2                          # expand @refs down to this nesting depth
                                        # (1 = referenced directly in the text)
# @token preceded by whitespace or start-of-string (skips emails, in-word @)
_AT_RE = re.compile(r"(?<!\S)@(\S+)")


def _parse_ref(token: str) -> tuple[str, int | None, int | None]:
    """Strip trailing punctuation; split off optional ':start[-end]' line range.

    Returns (path, start, end); start/end are 1-indexed, or None (whole file).
    Avoids treating Windows drive letters (e.g. 'C:\\path') as line specs by
    requiring ≥2-char path before ':' OR that the path contains '.' / '/' / '\\'.
    """
    token = token.rstrip(".,;:)")
    m = re.search(r":(\d+)(?:-(\d+))?$", token)
    if m:
        before = token[: m.start()]
        if len(before) > 2 or re.search(r"[./\\]", before):
            s = int(m.group(1))
            return before, s, int(m.group(2)) if m.group(2) else s
    return token, None, None


def _collect_refs(text: str, root: str, seen: set, depth: int,
                  blocks: list) -> None:
    """Append file blocks for all @path tokens in text (recursive)."""
    for m in _AT_RE.finditer(text):
        ref, lo1, hi1 = _parse_ref(m.group(1))
        resolved = os.path.realpath(os.path.join(root, ref))
        if resolved in seen or not os.path.isfile(resolved):
            continue  # graceful: missing/duplicate/cycle → leave token as-is
        try:
            if os.path.getsize(resolved) > _MAX_BYTES:
                continue
        except OSError:
            continue
        seen.add(resolved)
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                raw = f.readlines()
        except OSError:
            continue

        n = len(raw)
        lo = max(0, (lo1 - 1) if lo1 else 0)
        hi = min(n, hi1 if hi1 else n)
        content = "".join(raw[lo:hi])
        a_start, a_end = lo + 1, hi

        try:
            display = os.path.relpath(resolved, root)
        except ValueError:
            display = resolved   # absolute fallback (cross-drive on Windows)

        # Surface the nesting depth this file was found at (1 = direct ref). At
        # the ceiling, flag it when the file still holds @refs we won't expand,
        # so the reader knows the chain was truncated rather than empty.
        at_max = depth >= _MAX_DEPTH
        tag = f"depth {depth}"
        if at_max and _AT_RE.search("".join(raw)):
            tag += ", max depth reached — nested @refs not expanded"
        blocks.append(
            f"### {display} (lines {a_start}-{a_end}) [{tag}]\n"
            + content
        )
        # Recurse into refs inside this file; resolve relative to its own dir
        if not at_max:
            _collect_refs("".join(raw), os.path.dirname(resolved),
                          seen, depth + 1, blocks)


def expand_at_references(text: str, root: str = None) -> str:
    """Expand @path tokens in text, appending a '[Referenced files]' block.

    @path          — whole file (byte-capped at 256 KiB)
    @path:10-20    — lines 10-20 inclusive (1-indexed)
    @path:5        — line 5 only

    @tokens not preceded by whitespace are skipped (emails, in-word @ are safe).
    Missing files / unresolvable refs are left as plain text; no error raised.
    Depth-limited recursion (max depth 2) with cycle/dup protection; each block
    header carries the depth it was found at and a note when the cap truncated.
    Returns text unchanged when no resolvable refs are found.
    """
    if root is None:
        root = os.getcwd()
    blocks: list[str] = []
    _collect_refs(text, root, set(), 1, blocks)
    if not blocks:
        return text
    return text + "\n\n[Referenced files]\n" + "\n\n".join(blocks)
