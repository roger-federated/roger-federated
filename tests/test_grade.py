"""Tests for the user grade override (/grade) + the 10%-user-graded training gate.
Run with:  PYTHONPATH=src python -m pytest tests/test_grade.py

No model is loaded: the std_tools grade state is a couple module globals, and the trainer's
user-graded bookkeeping is stat-only (a `user_graded` sentinel file per run dir)."""
import roger.tools.std_tools as st
import roger.training.trainer as tr


def test_grade_state_machine():
    st.clear_grade()
    assert st.finish_score() is None                # None == nothing overridable
    assert st.set_user_grade(0.9) is None           # nothing pending -> no-op (can't grade out of band)

    st.finish(0.3)                                   # a finish opens the override window
    assert st.finish_score() == 0.3

    assert st.set_user_grade("nonsense") is None     # unparseable -> no-op, grade unchanged
    assert st.finish_score() == 0.3

    assert st.set_user_grade(0.9) == 0.9             # override sticks; window stays open for re-override
    assert st.finish_score() == 0.9
    assert st.set_user_grade(5) == 1.0               # clamped to [-1, 1]

    st.clear_grade()                                 # new task closes the window
    assert st.finish_score() is None


def test_user_grade_shortfall(tmp_path, monkeypatch):
    # Build N run dirs, mark some user-graded, and check the shortfall (the 10%-or-1 rule's seam).
    def dirs(n):
        out = []
        for i in range(n):
            d = tmp_path / f"run{i}"
            d.mkdir(exist_ok=True)
            out.append(str(d))
        return out

    eight = dirs(8)
    monkeypatch.setattr(tr, "_list_unconsumed", lambda batch: eight)
    assert tr.user_grade_shortfall() == 1            # batch of 8 needs one user-graded run
    (tmp_path / "run0" / "user_graded").touch()
    assert tr.user_grade_shortfall() == 0            # one suffices

    eleven = dirs(11)                                # run0 stays graded from above
    monkeypatch.setattr(tr, "_list_unconsumed", lambda batch: eleven)
    assert tr.user_grade_shortfall() == 1            # ceil(0.10 * 11) == 2, minus run0
    (tmp_path / "run1" / "user_graded").touch()
    assert tr.user_grade_shortfall() == 0

    # Fewer than 2 runs never trains, so there's nothing to nudge.
    monkeypatch.setattr(tr, "_list_unconsumed", lambda batch: eleven[:1])
    assert tr.user_grade_shortfall() == 0
