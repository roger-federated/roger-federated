---
name: lean
description: Write and check Lean 4 / mathlib proofs and programs (build loop, reading goal state, searching mathlib, axiom-checking, never leaving sorry). Use for theorem proving or any Lean formalization task.
---

Lean is a *local* toolchain: write `.lean` files, run `lake build`, read the goals and errors
it prints, iterate. The build is the source of truth; a proof is done only when it builds with
no errors, no `sorry`/`admit`, and `#print axioms` shows nothing unexpected. Treat a clean build
as the verifiable success signal and a failing one as the exact thing to fix next.

Toolchain: `elan` manages Lean versions; `lake` is the build tool. A project has a
`lakefile.toml` (or `.lean`) and a `lean-toolchain` pinning the version. If it depends on
mathlib, run `lake exe cache get` *before* `lake build` to download prebuilt `.olean`s; building
mathlib from source otherwise takes hours. `lake update` refreshes dependencies (only when
asked; it can move versions).

Work in stages, smallest first: draft the skeleton (`theorem foo : P := by sorry`) so it
type-checks, then close each `sorry` one at a time. After each step `lake build` (or
`lake build Some.Module` to scope it), read the output top-down, and fix the *first* error
first; later ones are usually cascades. This compiler-guided repair beats re-drafting the whole
proof from scratch when one tactic fails.

Read the state, do not guess at it. Lean tells you exactly where you are:
- Unsolved goals print at the failing tactic, hypotheses above the `⊢` and the target below it.
  Prove *that* goal, not what you assumed it was.
- `#check e`, `#print name`, `#eval e` are throwaway probes; leave none behind.
- Type-mismatch errors give "expected ... / got ..." — read both before changing anything.

Search mathlib before reinventing a lemma. From inside a proof, `exact?` / `apply?` / `rw?` /
`simp?` ask Lean to find or spell out the step; `#find` and loogle/leansearch search by shape or
name. Follow mathlib naming (`add_comm`, `mul_pos`, lowercase_snake by conclusion). Prefer an
existing lemma over a hand-rolled one.

Tactics, minimally: `intro`, `exact`, `apply`, `rw`, `simp`, `ring`, `omega`, `linarith`,
`constructor`, `rcases`/`obtain`, `induction`. Reach for the automation (`simp`/`omega`/
`linarith`/`ring`/`aesop`) before manual term-mode; it is shorter and more robust. Once a goal
closes, golf it (shorter, clearer, fewer custom lemmas) only if the build stays green.

Two things to never do. Never paper over a gap: `sorry` makes a file "build" while proving
nothing, and a proof that compiles via an unexpected axiom is not closed, so always confirm with
`#print axioms <name>`. And never weaken or rewrite the statement just to make it pass; the
theorem you were asked to prove is fixed, exactly like not editing a test to make it green. Match
the file's existing style (term vs tactic mode, naming, structure) rather than imposing your own.
