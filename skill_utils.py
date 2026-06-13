"""skill_utils.py — instruction-file loading and skill progressive-disclosure.

Conventions:
  AGENTS.md / CLAUDE.md              — freeform project instructions, prepended to system prompt
  .roger/.agents/.claude/skills/*.md — per-skill files with YAML frontmatter (name, description)
                              + a markdown body; only the catalog is shown up front;
                              model fetches the body on demand via load_skill(name).
"""

import os, glob
from path_utils import expand_at_references

# ---------------------------------------------------------------------------
# Instruction files
# ---------------------------------------------------------------------------

def load_instructions(root: str) -> str:
    """Return the content of the first AGENTS.md or CLAUDE.md found in root.

    Tries AGENTS.md first (vendor-neutral cross-agent standard), then CLAUDE.md.
    Returns "" when neither exists; the caller decides whether to append to sys_content.
    """
    for name in ("AGENTS.md", "CLAUDE.md"):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            try:
                text = open(path, encoding="utf-8").read().strip()
            except OSError:
                continue
            if text:
                return f"## Project instructions ({name})\n{text}"
    return ""


# ---------------------------------------------------------------------------
# Skill discovery + progressive-disclosure loader
# ---------------------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse minimal YAML-style frontmatter delimited by '---' lines.

    Returns (meta_dict, body_str).  Only handles simple 'key: value' lines;
    no lists, no nested YAML — sufficient for name/description fields.
    """
    lines = text.splitlines()
    # Need at least opening --- ... closing --- to have frontmatter
    if len(lines) < 2 or lines[0].strip() != "---":
        return {}, text
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    meta = {}
    for line in lines[1:end]:
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    body = "\n".join(lines[end + 1:]).strip()
    return meta, body


def discover_skills(root: str) -> list[dict]:
    """Discover skill files under root, deduplicating by resolved path.

    Searches three folder names (.roger/skills, .agents/skills, .claude/skills) and two layouts:
      nested: <folder>/<name>/SKILL.md  — name from frontmatter or parent dir
      flat:   <folder>/<name>.md        — name from frontmatter or file stem

    Each record: {name, description, body, path}.
    Skips files with no usable description (would make a useless catalog entry).
    """
    _FOLDERS = (".roger", ".agents", ".claude")  # .roger is canonical; others for cross-agent compat
    seen, skills = set(), []
    for folder in _FOLDERS:
        base = os.path.join(root, folder, "skills")
        patterns = [
            (os.path.join(base, "*", "SKILL.md"), "nested"),
            (os.path.join(base, "*.md"),           "flat"),
        ]
        for pattern, layout in patterns:
            for path in sorted(glob.glob(pattern)):
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    text = open(path, encoding="utf-8").read()
                except OSError:
                    continue
                meta, body = _parse_frontmatter(text)
                desc = meta.get("description", "").strip()
                if not desc:
                    continue
                # name: frontmatter > dir name (nested) > file stem (flat)
                if layout == "nested":
                    fallback = os.path.basename(os.path.dirname(path))
                else:
                    fallback = os.path.splitext(os.path.basename(path))[0]
                name = meta.get("name", "").strip() or fallback
                # Expand @path refs relative to the skill's own directory
                body = expand_at_references(body, os.path.dirname(path))
                skills.append({"name": name, "description": desc, "body": body, "path": path})
    return skills


def make_skill_loader(skills: list[dict]) -> tuple[str, callable]:
    """Build a terse catalog string + load_skill closure (mirrors make_tool_loader).

    Returns (catalog_text, load_skill) where:
      catalog_text  — one '- name: description' line per skill
      load_skill    — callable(name: str) -> str; returns the skill body on demand
    """
    index = {s["name"]: s["body"] for s in skills}
    catalog_text = "\n".join(f"- {s['name']}: {s['description']}" for s in skills)

    def load_skill(name: str) -> str:
        """Return the full body of a skill by name.
        Args:
            name: skill name as listed in the catalog
        Returns: skill instructions as a markdown string
        """
        if name not in index:
            return f"Unknown skill '{name}'. Available: {', '.join(index)}"
        return index[name]

    return catalog_text, load_skill
