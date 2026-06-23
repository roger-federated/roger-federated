"""skill_utils.py — instruction-file loading and skill progressive-disclosure.

Conventions:
  AGENTS.md / CLAUDE.md              — freeform project instructions, prepended to system prompt
  Memory (two tiers, both under ~/.roger/memory/, never in the project):
    ~/.roger/memory/memory.md  — user-level facts (preferences, identity), cross-project
    ~/.roger/memory/<key>.md   — facts specific to the current project (key = its abspath
                                 with separators dashed, like Claude Code)
  Skills (per-skill .md with YAML frontmatter + body; only the catalog is shown up front,
  body fetched on demand via load_skill(name)) are discovered in ~/.roger/skills (global) and
  the project's .agents/skills and .claude/skills.
"""

import os, re, glob
from roger.agency.path_utils import expand_at_references, state_dir

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
                with open(path, encoding="utf-8") as f:
                    text = f.read().strip()
            except OSError:
                continue
            if text:
                return f"## Project instructions ({name})\n{text}"
    return ""


def project_mem_file(root: str) -> str:
    """Per-project memory file under ~/.roger/memory/, named by the project's abspath with path
    separators dashed out (e.g. C:\\…\\newventure → C--…-newventure.md), mirroring Claude Code."""
    key = re.sub(r"[\\/:]", "-", os.path.abspath(root))
    return os.path.join(state_dir(), "memory", f"{key}.md")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def load_memory(root: str) -> str:
    """Write protocol + current contents of both memory tiers (global + this project)."""
    g = os.path.join(state_dir(), "memory", "memory.md")
    p = project_mem_file(root)
    block = ("## Persistent memory\nUpdate at the end of each task: user-level facts "
             f"(preferences, identity) → {g}; facts specific to this project → {p}.")
    return (block
            + f"\n\nGlobal memory:\n{_read(g) or '(none yet)'}"
            + f"\n\nProject memory:\n{_read(p) or '(none yet)'}")


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
    """Discover skill files, deduplicating by resolved path.

    Searches the bundled defaults shipped with roger plus the global ~/.roger/skills and the
    project's .agents/skills and .claude/skills (cross-agent compat), in two layouts:
      nested: <base>/<name>/SKILL.md  — name from frontmatter or parent dir
      flat:   <base>/<name>.md        — name from frontmatter or file stem

    Each record: {name, description, body, path}.
    Skips files with no usable description (would make a useless catalog entry).
    """
    # Bundled defaults ship as package-data under roger/skills/ (this file is roger/agency/).
    # Listed first = lowest priority: a same-named user/project skill is later in the list and
    # so wins in make_skill_loader's name->body index, letting users shadow a default.
    bundled = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
    bases = (
        bundled,                                      # shipped defaults (overridable)
        os.path.join(state_dir(), "skills"),          # global, canonical
        os.path.join(root, ".agents", "skills"),      # project, cross-agent
        os.path.join(root, ".claude", "skills"),      # project, cross-agent
    )
    seen, skills = set(), []
    for base in bases:
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
                    with open(path, encoding="utf-8") as f:
                        text = f.read()
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
    """Build a terse catalog string + load_skill closure (mirrors _make_tool_loader).

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
