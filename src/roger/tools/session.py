"""Per-agent tool state (ToolSession).

Every agent — the main rollout and each spawned sub-agent — owns one ToolSession holding the
mutable state the tools touch: background shell jobs, write backups (for /revert), and the
self-grade window, plus the prompt/policy config. Isolating this per agent is what makes
concurrent sub-agents safe: this state used to live in module globals that a nested rollout
clobbered (job registry cleared, grade overwritten, backups reset). Tools are bound to a session
by std_tools.get_standard_tools(session), which returns handler closures over that instance.

OOP is warranted here precisely because this is isolated mutable state with a small, cohesive
interface — the grade window, the job registry, and the backup list — that every tool and the
rollout loop read and mutate.
"""
import os, re, shutil
from dataclasses import dataclass, field
from typing import Callable


def _default_prompt(question: str) -> str:
    try:
        return input(question + " ")
    except EOFError:
        return "(no response — user input unavailable)"
    except KeyboardInterrupt:
        return "(user cancelled)"


@dataclass
class ToolSession:
    # Config (a child copies these from its parent; overridable without touching the parent)
    prompt_backend: Callable | None = _default_prompt   # confirm/ask prompts route through this
    policy_file: str = "command_policy.txt"             # run_command guardrail policy path
    # Per-agent mutable state
    jobs: dict = field(default_factory=dict)            # id -> {proc, future, command, reported, outfile}
    job_seq: int = 0
    backups: list = field(default_factory=list)         # (orig_path, backup_path) for end-of-rollout revert
    grade: float | None = None                          # pending self/user grade for the current segment
    grade_open: bool = False                            # /grade window for the current user turn

    def __post_init__(self):
        if self.prompt_backend is None:                 # callers may pass None to mean "use the default"
            self.prompt_backend = _default_prompt

    # --- self-grade window (was std_tools module state) ---
    def grade_value(self) -> float | None:
        """Pending grade, or None when nothing has been graded yet / window is closed."""
        return self.grade

    def set_grade(self, score) -> float | None:
        """Set the grade; returns the clamped value, or None on parse error."""
        try:
            self.grade = max(-1.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return None
        return self.grade

    def record_grade(self, score: float = 0.0) -> str:
        """Handler for the silent grade nudge — sets grade_value() and returns '(done)'."""
        self.set_grade(score)
        return "(done)"

    def clear_grade(self) -> None:
        """Close the grade window at a new task boundary."""
        self.grade = None

    def gradeable(self) -> bool:
        """True when the current user turn follows a task and /grade is active."""
        return self.grade_open

    def set_gradeable(self, value: bool) -> None:
        """Open/close the /grade window (called by the rollout around _await_user_turn)."""
        self.grade_open = value

    # --- background jobs (rollout-facing; run_command in shell_tools fills `jobs`) ---
    def drain_finished_jobs(self) -> list[tuple[str, str]]:
        """(id, output) for jobs that finished since the last drain, marking them reported."""
        out = []
        for jid, job in self.jobs.items():
            if job["future"].done() and not job["reported"]:
                job["reported"] = True
                out.append((jid, job["future"].result()))
        return out

    def pending_jobs(self) -> list:
        """Futures of background commands still running."""
        return [j["future"] for j in self.jobs.values() if not j["future"].done()]

    def terminate_jobs(self) -> None:
        """Kill all still-running background commands and clear the registry (rollout cleanup)."""
        for job in self.jobs.values():
            if not job["future"].done():
                job["proc"].kill()
            try: os.remove(job["outfile"])
            except OSError: pass
        self.jobs.clear()

    # --- write backups / revert (was std_tools module state) ---
    def pending_backups(self) -> list[tuple[str, str]]:
        """Current (original_path, backup_path) pairs awaiting a revert decision; empty when nothing changed."""
        return list(self.backups)

    def apply_revert(self, answer: str) -> int:
        """Revert backed-up files per `answer` ('all' / '1,3' / 'none'); return the count reverted.
        Only the reverted entries are popped — a partial revert keeps the rest available for a later
        offer. 'none' discards everything; an answer with no recognised index is a no-op (doesn't
        clear), so an accidental ghost-text accept can't wipe the list."""
        answer = (answer or "").strip().lower()
        if answer == "none":
            self.backups.clear(); return 0
        if answer in ("", "all"):
            indices = set(range(len(self.backups)))
        else:
            # Keep only integer tokens (split on commas/whitespace); ignore anything else.
            indices = {int(t) - 1 for t in re.split(r"[,\s]+", answer) if t.isdigit()}
            if not indices:
                return 0
        n_reverted = 0
        kept: list[tuple[str, str]] = []
        for idx, (orig, bak) in enumerate(self.backups):
            if idx in indices:
                try: shutil.copy2(bak, orig); n_reverted += 1
                except OSError: kept.append((orig, bak))   # restore failed → keep so the user can retry
            else:
                kept.append((orig, bak))                   # not selected → still pending
        self.backups[:] = kept
        return n_reverted
