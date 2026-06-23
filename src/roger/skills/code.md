---
name: code
description: Write, debug, and test code changes well (match existing style, smallest diff, reproduce-isolate-fix-verify). Use for any non-trivial coding, debugging, or testing.
---

A project's own AGENTS.md/CLAUDE.md overrides these defaults where they differ.

Understand before changing: read the target file and its neighbours. Match the surrounding
style (naming, idioms, comment density, error handling). Don't introduce a new pattern when
the codebase already has one. Never edit a file you have not read this session.

Minimal change: make the smallest diff that solves the actual problem. No speculative
abstraction, no future-proofing, no drive-by refactors of unrelated code. Don't delete code
or comments you weren't asked to touch.

Debugging loop:
1. Reproduce: get a concrete command or test that fails reliably. No repro, no fix.
2. Isolate: narrow it down (bisect, add logging, shrink the input) until the cause is local.
3. Hypothesize: form one specific theory of the bug.
4. Fix: make the smallest change that addresses the cause, not the symptom.
5. Verify: confirm the repro now passes and the surrounding behaviour still works.

Tests: for a bug, first write a failing test that captures it, then fix until green; that
proves the fix and guards against regressions. For a feature, add or extend tests covering
the new behaviour and its edge cases. Run the existing suite before declaring done. Never
edit a test just to make it pass: if a test looks wrong, work out whether the test or the
code is at fault and say so, rather than silently changing the assertion to match buggy
output.

Verify for real: actually execute the code and tests via `run_command`. Don't claim success
from reading the diff. If something fails or you skipped a step, say so plainly with the
output; never report green when you didn't see green.
