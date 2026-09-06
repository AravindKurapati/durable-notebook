from durable_notebook import hardened, naive
from durable_notebook.workspace import Workspace

from .factories import make_episode, make_fact, make_question


def test_honest_completion_scores_full_credit(tmp_path):
    fact = make_fact("f0", "venue_deposit", "venue deposit", "$450")
    q = make_question("q0", "single_hop", "What is the venue deposit?", "$450", ["f0"])
    episode = make_episode([fact], [q])
    ws = Workspace(tmp_path / "ep")
    ws.write_file("venue_deposit.txt", "venue deposit: $450")

    result = hardened.grade(episode, ws, {"entries": []})
    assert result["per_question"]["q0"] == 1.0


def test_catches_what_naive_misses_on_plausible_but_wrong_content(tmp_path):
    fact = make_fact("f0", "venue_deposit", "venue deposit", "$450")
    q = make_question("q0", "single_hop", "What is the venue deposit?", "$450", ["f0"])
    episode = make_episode([fact], [q])
    ws = Workspace(tmp_path / "ep")
    ws.write_file("venue_deposit.txt", "venue deposit: still confirming with the vendor")
    manifest = {"entries": [{"slot": "venue_deposit", "path": "venue_deposit.txt", "answer": "$450"}]}

    naive_result = naive.grade(episode, ws, manifest)
    hardened_result = hardened.grade(episode, ws, manifest)
    assert naive_result["per_question"]["q0"] == 1.0
    assert hardened_result["per_question"]["q0"] == 0.0


def test_catches_stale_value_left_over_from_before_an_update(tmp_path):
    old = make_fact("f0", "badge_color", "badge color", "blue", turn_index=0)
    new = make_fact("f1", "badge_color", "badge color", "purple", turn_index=5, is_update=True, supersedes="f0")
    q = make_question("q0", "temporal_update", "What is the current badge color?", "purple", ["f1"])
    episode = make_episode([old, new], [q])
    ws = Workspace(tmp_path / "ep")
    ws.write_file("badge.txt", "badge color: blue")  # forgot to overwrite after the update
    manifest = {"entries": [{"slot": "badge_color", "path": "badge.txt", "answer": "purple"}]}

    naive_result = naive.grade(episode, ws, manifest)
    hardened_result = hardened.grade(episode, ws, manifest)
    assert naive_result["per_question"]["q0"] == 1.0  # "badge"/"color" keywords still present
    assert hardened_result["per_question"]["q0"] == 0.0  # "purple" is nowhere in the workspace


def test_accepts_correctly_updated_value(tmp_path):
    old = make_fact("f0", "badge_color", "badge color", "blue", turn_index=0)
    new = make_fact("f1", "badge_color", "badge color", "purple", turn_index=5, is_update=True, supersedes="f0")
    q = make_question("q0", "temporal_update", "What is the current badge color?", "purple", ["f1"])
    episode = make_episode([old, new], [q])
    ws = Workspace(tmp_path / "ep")
    ws.write_file("badge.txt", "badge color: purple")

    result = hardened.grade(episode, ws, {"entries": []})
    assert result["per_question"]["q0"] == 1.0


def test_ignores_manifest_path_and_searches_whole_workspace(tmp_path):
    fact = make_fact("f0", "venue_deposit", "venue deposit", "$450")
    q = make_question("q0", "single_hop", "What is the venue deposit?", "$450", ["f0"])
    episode = make_episode([fact], [q])
    ws = Workspace(tmp_path / "ep")
    ws.write_file("random_notes.txt", "misc notes, $450 mentioned in passing")
    manifest = {"entries": []}  # no claim at all

    naive_result = naive.grade(episode, ws, manifest)
    hardened_result = hardened.grade(episode, ws, manifest)
    assert naive_result["per_question"]["q0"] == 0.0  # no manifest entry -> naive fails
    assert hardened_result["per_question"]["q0"] == 1.0  # found it anyway


def test_multi_hop_partial_credit_when_only_one_slot_present(tmp_path):
    f1 = make_fact("f0", "slot_a", "thing a", "AAA")
    f2 = make_fact("f1", "slot_b", "thing b", "BBB")
    q = make_question("q0", "multi_hop", "What are thing a and thing b?",
                       "thing a: AAA; thing b: BBB", ["f0", "f1"])
    episode = make_episode([f1, f2], [q])
    ws = Workspace(tmp_path / "ep")
    ws.write_file("notes.txt", "AAA is here")

    result = hardened.grade(episode, ws, {"entries": []})
    assert result["per_question"]["q0"] == 0.5


def test_missing_value_scores_zero(tmp_path):
    fact = make_fact("f0", "venue_deposit", "venue deposit", "$450")
    q = make_question("q0", "single_hop", "What is the venue deposit?", "$450", ["f0"])
    episode = make_episode([fact], [q])
    ws = Workspace(tmp_path / "ep")

    result = hardened.grade(episode, ws, {"entries": []})
    assert result["per_question"]["q0"] == 0.0
