from durable_notebook.tools import list_files, read_file, submit_manifest, write_file
from durable_notebook.workspace import Workspace


def _state(tmp_path, name="ep"):
    return {"dn_workspace": Workspace(tmp_path / name)}


def test_write_read_list_roundtrip(tmp_path):
    state = _state(tmp_path)
    result = write_file("notes.txt", "hello", state)
    assert "wrote" in result
    assert read_file("notes.txt", state) == "hello"
    assert list_files(state) == "notes.txt"


def test_list_files_empty_workspace_message(tmp_path):
    state = _state(tmp_path)
    assert list_files(state) == "(workspace is empty)"


def test_read_missing_file_returns_error_string_not_exception(tmp_path):
    state = _state(tmp_path)
    result = read_file("nope.txt", state)
    assert result.startswith("error:")


def test_path_traversal_returns_error_string_not_exception(tmp_path):
    state = _state(tmp_path)
    result = write_file("../escape.txt", "x", state)
    assert result.startswith("error:")


def test_two_episode_states_are_isolated(tmp_path):
    state_a = _state(tmp_path, "ep-a")
    state_b = _state(tmp_path, "ep-b")
    write_file("f.txt", "only in a", state_a)
    assert list_files(state_b) == "(workspace is empty)"


def test_submit_manifest_valid_entries_updates_state(tmp_path):
    state = _state(tmp_path)
    state["dn_qa_shown"] = True  # submission is only legal once asked
    entries = [{"slot": "a", "path": "a.txt", "answer": "x"}]
    result = submit_manifest(entries, state)
    assert result == "manifest received, thank you."
    assert state["dn_manifest_submitted"] is True
    assert state["dn_manifest"] == {"entries": entries}


def test_submit_manifest_malformed_shape_does_not_mark_submitted(tmp_path):
    state = _state(tmp_path)
    result = submit_manifest([{"slot": "a"}], state)  # missing path/answer
    assert result.startswith("error:")
    assert "dn_manifest_submitted" not in state


def test_submit_manifest_accepts_empty_entries_as_a_valid_but_vacuous_shape(tmp_path):
    state = _state(tmp_path)
    state["dn_qa_shown"] = True  # submission is only legal once asked
    result = submit_manifest([], state)
    assert result == "manifest received, thank you."
    assert state["dn_manifest"] == {"entries": []}


# -- premature submission ------------------------------------------------
#
# submit_manifest used to end the rollout whenever it was called, including
# on turn 1 -- before the episode's facts had been delivered and before the
# QA turn had ever been shown. Measured on a real 35-episode gpt-oss-20b
# run (2026-08-17): 18 of 30 scored episodes never referenced any fact past
# turn 6 of a 12-turn episode, and 5 referenced nothing at all. That made
# Gate 1 measure "how far did the policy get before quitting" rather than
# "does the task require persistence", and every question type showed a
# negative visible-minus-compacted gap as a result.
#
# It is also a degenerate strategy GRPO would be free to find: end the
# episode immediately and skip the hard part entirely. The environment has
# to refuse it rather than rely on the policy's good behaviour.


def test_submit_before_questions_asked_is_rejected(tmp_path):
    state = _state(tmp_path)
    state["dn_qa_shown"] = False
    result = submit_manifest([{"slot": "a", "path": "a.txt", "answer": "x"}], state)
    assert result.startswith("error:")
    assert not state.get("dn_manifest_submitted")
    assert state.get("dn_manifest") is None


def test_submit_after_questions_asked_is_accepted(tmp_path):
    state = _state(tmp_path)
    state["dn_qa_shown"] = True
    entries = [{"slot": "a", "path": "a.txt", "answer": "x"}]
    result = submit_manifest(entries, state)
    assert not result.startswith("error:")
    assert state["dn_manifest_submitted"] is True
    assert state["dn_manifest"] == {"entries": entries}


def test_rejected_early_submit_does_not_end_the_rollout(tmp_path):
    """The rejection must leave the episode running so turn delivery
    continues -- otherwise refusing the call is no better than accepting
    it."""
    state = _state(tmp_path)
    state["dn_qa_shown"] = False
    submit_manifest([{"slot": "a", "path": "a.txt", "answer": "x"}], state)
    assert not state.get("dn_manifest_submitted")
    state["dn_qa_shown"] = True
    result = submit_manifest([{"slot": "a", "path": "a.txt", "answer": "x"}], state)
    assert not result.startswith("error:")
    assert state["dn_manifest_submitted"] is True


def test_early_submit_error_tells_the_model_what_to_do(tmp_path):
    state = _state(tmp_path)
    state["dn_qa_shown"] = False
    result = submit_manifest([{"slot": "a", "path": "a.txt", "answer": "x"}], state)
    assert "question" in result.lower()


def test_shape_validation_still_runs_before_the_qa_gate(tmp_path):
    """A malformed manifest submitted late is still a shape error, and a
    well-formed one submitted early is still a timing error -- neither
    check may shadow the other."""
    state = _state(tmp_path)
    state["dn_qa_shown"] = True
    result = submit_manifest([{"slot": "a"}], state)
    assert result.startswith("error:")
    assert not state.get("dn_manifest_submitted")
