---
name: skill-creator
description: Author or improve a roger skill (a .md file with YAML frontmatter). Use when asked to create, add, package, or refine a skill, or when a recurring procedure is worth capturing for reuse.
---

A skill is reusable procedural knowledge the agent loads on demand. Only the `description` is
shown up front in the catalog; the body is fetched via `load_skill(name)` when the description
looks relevant. A skill earns its place only when it captures a *procedure*, especially a
fragile-without-it one, rather than a thin wrapper over a tool or MCP server that already exists.

What makes a good skill (apply these, they are why skills work):
- Generalize, don't overfit. Write for the many future runs with varied prompts, not the one
  task in front of you. Prefer broad patterns over a brittle step-by-step that breaks on the
  next variation.
- Stay lean. Every line must pull its weight; cut anything the base model already knows or that
  only helped on one example.
- Explain the reasoning, not just the rule. Say *why* a step matters so the model can adapt when
  reality differs, instead of shouting ALWAYS/NEVER at it.

Where to write one:
- User, all projects: `~/.roger/skills/<name>.md`
- This project only: `<project>/.agents/skills/<name>.md` (or `.claude/skills/...`)
- Bundled defaults ship with roger and are read-only; a user/project skill of the same name
  shadows the default.

Two layouts:
- flat `<name>.md`
- nested `<name>/SKILL.md` when the skill bundles its own files. Put those in `scripts/`
  (runnable code), `references/` (docs loaded only when needed), or `assets/` (templates), and
  point to them from the body with `@path` relative to the skill dir. This keeps the body short:
  depth lives in the bundled file, pulled in on demand.

Format:
```markdown
---
name: kebab-case-name
description: <what it does AND when to use it, in one cue-rich sentence>
---

<imperative, procedural body: the HOW, with a concrete example or two>
```
Only `name` and `description` are parsed from the frontmatter; a skill with no description is
skipped entirely.

The description is the whole trigger, so write it to be found. State both what the skill covers
and the situations that should invoke it, and lean slightly pushy: name the contexts, file
types, or phrasings a user might use even when they don't ask for the skill by name. A vague
topic label undertriggers and the skill never loads.

Keep the body focused and imperative, and show a realistic example rather than only describing
one. The new skill appears in the catalog on the next reload; confirm with `load_skill(name)`.
