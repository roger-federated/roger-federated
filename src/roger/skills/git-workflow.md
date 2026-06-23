---
name: git-workflow
description: Local git practice (atomic commits, clear messages, branching, conflict resolution, safe history ops). Use when committing, branching, or repairing git state.
---

This is *local* git hygiene. Platform actions (opening PRs, issues, reviews) are a GitHub
CLI or MCP concern, not this skill.

Look before you act: `git status`, `git diff`, `git log --oneline -n 20`.

Atomic commits: one logical change per commit. Stage selectively with `git add -p`; split
unrelated changes into separate commits rather than one mixed blob.

Messages: imperative summary line (~50-72 chars), then a blank line, then a body explaining
*why* when it is not obvious from the diff. Don't restate the diff.

Branching: don't commit experimental or risky work straight onto `main`/`master`; branch
first with `git switch -c <branch>`.

Before anything destructive (`reset --hard`, `push --force`, `checkout -- <file>`,
`clean -fd`): pause, confirm intent, and prefer the safe alternative. Use `git revert` over
`reset` on shared history, `--force-with-lease` over `--force`, `git stash` over discarding.

Conflicts: `git status` lists the conflicted files; resolve each, `git add`, then continue
the rebase/merge. If unsure, `--abort` and reassess rather than guessing.

Hygiene: never commit secrets or large build artifacts; check `.gitignore` first. Prefer
`git pull --rebase` to keep history linear and avoid merge-commit noise.
