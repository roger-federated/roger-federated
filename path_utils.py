"""path_utils.py — @path reference expansion and shared line-gutter primitive.

Public API:
  gutter(text, start)              — right-aligned line-number rendering
  expand_at_references(text, root) — expand @path tokens; append referenced blocks

Both std_tools.read_file and retrieval.format_context use gutter() so all
file-content surfaces look identical to the model.
"""
import os, re

_MAX_BYTES = 262144                     # mirrors retrieval.build_index max_bytes
# @token preceded by whitespace or start-of-string (skips emails, in-word @)
_AT_RE = re.compile(r"(?<!\S)@(\S+)")


def gutter(text: str, start: int = 1) -> str:
    """Render text with right-aligned 1-indexed line numbers '   N | line'."""
    return "\n".join(
        f"{start + i:>4} | {line}"
        for i, line in enumerate(text.rstrip("\n").split("\n"))
    )


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
    """Append gutter-rendered file blocks for all @path tokens in text (recursive)."""
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

        blocks.append(
            f"### {display} (lines {a_start}-{a_end})\n"
            + gutter(content, a_start)
        )
        # Recurse into refs inside this file; resolve relative to its own dir
        if depth < 5:
            _collect_refs("".join(raw), os.path.dirname(resolved),
                          seen, depth + 1, blocks)


def expand_at_references(text: str, root: str = None) -> str:
    """Expand @path tokens in text, appending a '[Referenced files]' block.

    @path          — whole file (byte-capped at 256 KiB)
    @path:10-20    — lines 10-20 inclusive (1-indexed)
    @path:5        — line 5 only

    @tokens not preceded by whitespace are skipped (emails, in-word @ are safe).
    Missing files / unresolvable refs are left as plain text; no error raised.
    Depth-limited recursion (5 levels) with cycle/dup protection.
    Returns text unchanged when no resolvable refs are found.
    """
    if root is None:
        root = os.getcwd()
    blocks: list[str] = []
    _collect_refs(text, root, set(), 0, blocks)
    if not blocks:
        return text
    return text + "\n\n[Referenced files]\n" + "\n\n".join(blocks)
