"""Tests for Gate 1's answer scorer.

Gate 1 measures whether the policy KNOWS a fact, not whether it obeyed the
manifest protocol. Those come apart in practice: against a real model the
scorer was returning 0 for multi-hop questions the model had answered
correctly, because the model merged the two asked-about keys into a single
invented slot name. Every such question landed in the "visible" bucket as a
miss, which is what dragged visible accuracy down to 0.381 and made the gate
read FAIL when the underlying behaviour was fine.

Run from the repo root: pytest eval/test_gate1_scoring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "durable_notebook"))

from durable_notebook.generator import Episode, Fact, Question  # noqa: E402

import gate1_persistence_ablation as gate1  # noqa: E402


def _episode(facts, questions):
    return Episode(
        episode_id="test-ep", seed=0, scenario="test", split="train",
        turns=(), questions=tuple(questions), facts=tuple(facts),
    )


def _fact(id, slot, label, value, turn_index=0, is_update=False, supersedes=None):
    return Fact(id=id, scenario="test", slot=slot, label=label, value=value,
                turn_index=turn_index, is_update=is_update, supersedes=supersedes)


def _multi_hop_episode():
    f1 = _fact("f0", "sponsor_budget", "sponsor budget", "$1200")
    f2 = _fact("f1", "badge_color", "badge color", "yellow")
    q = Question(
        id="q-multi-sponsor_budget-badge_color", kind="multi_hop",
        text="What are the sponsor budget (key: sponsor_budget) and the badge color (key: badge_color)?",
        answer="sponsor budget: $1200; badge color: yellow",
        supporting_fact_ids=("f0", "f1"),
    )
    return _episode([f1, f2], [q]), q


# -- the well-behaved baseline must keep working ------------------------


def test_exact_per_slot_entries_score_full_credit():
    episode, q = _multi_hop_episode()
    manifest = {"entries": [
        {"slot": "sponsor_budget", "path": "a.txt", "answer": "$1200"},
        {"slot": "badge_color", "path": "b.txt", "answer": "yellow"},
    ]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 1.0


def test_wrong_values_in_exact_entries_score_zero():
    episode, q = _multi_hop_episode()
    manifest = {"entries": [
        {"slot": "sponsor_budget", "path": "a.txt", "answer": "$99"},
        {"slot": "badge_color", "path": "b.txt", "answer": "green"},
    ]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 0.0


def test_partial_credit_when_one_slot_right():
    episode, q = _multi_hop_episode()
    manifest = {"entries": [
        {"slot": "sponsor_budget", "path": "a.txt", "answer": "$1200"},
        {"slot": "badge_color", "path": "b.txt", "answer": "green"},
    ]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 0.5


# -- the actual bug -----------------------------------------------------


def test_merged_slot_entry_with_correct_values_gets_credit():
    """VERBATIM from a real gpt-oss-120b rollout (2026-08-17): the model
    concatenated both keys into one slot and answered in prose. Both values
    are right, so Gate 1 -- which is asking whether the model knew them --
    must not score this 0."""
    episode, q = _multi_hop_episode()
    manifest = {"entries": [{
        "slot": "sponsor_budget_and_badge_color",
        "path": "n/a",
        "answer": "Sponsor budget is $1200 and badge color is yellow.",
    }]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 1.0


def test_merged_slot_entry_missing_one_value_gets_partial_credit():
    episode, q = _multi_hop_episode()
    manifest = {"entries": [{
        "slot": "sponsor_budget_and_badge_color",
        "path": "n/a",
        "answer": "Sponsor budget is $1200; I don't recall the badge color.",
    }]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 0.5


def test_merged_slot_entry_with_wrong_values_still_scores_zero():
    """The fallback must not become 'any prose scores'. Wrong is wrong."""
    episode, q = _multi_hop_episode()
    manifest = {"entries": [{
        "slot": "sponsor_budget_and_badge_color",
        "path": "n/a",
        "answer": "Sponsor budget is $99 and badge color is green.",
    }]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 0.0


def test_dont_know_answer_scores_zero():
    """The honest refusal a tool-less model gives for a compacted fact --
    it must stay a miss, or the compacted bucket inflates and the gate
    stops measuring anything."""
    episode, q = _multi_hop_episode()
    manifest = {"entries": [
        {"slot": "sponsor_budget", "path": "n/a", "answer": "I don't have that information."},
        {"slot": "badge_color", "path": "n/a", "answer": "I don't have that information."},
    ]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 0.0


def test_unrelated_merged_entry_does_not_leak_credit():
    """An entry whose slot name has nothing to do with the asked slot must
    not be consulted -- otherwise one big dump-everything entry scores every
    question and the visible/compacted contrast collapses."""
    episode, q = _multi_hop_episode()
    manifest = {"entries": [{
        "slot": "unrelated_notes",
        "path": "n/a",
        "answer": "$1200 yellow",
    }]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 0.0


def test_single_hop_exact_entry_unaffected():
    f = _fact("f0", "venue_deposit", "venue deposit", "$450")
    q = Question(id="q-single-venue_deposit", kind="single_hop",
                 text="What is the venue deposit? (key: venue_deposit)",
                 answer="$450", supporting_fact_ids=("f0",))
    episode = _episode([f], [q])
    manifest = {"entries": [{"slot": "venue_deposit", "path": "a.txt", "answer": "$450"}]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 1.0


def test_case_and_whitespace_insensitive():
    f = _fact("f0", "badge_color", "badge color", "Yellow")
    q = Question(id="q-single-badge_color", kind="single_hop",
                 text="What is the badge color? (key: badge_color)",
                 answer="Yellow", supporting_fact_ids=("f0",))
    episode = _episode([f], [q])
    manifest = {"entries": [{"slot": "badge_color", "path": "a.txt", "answer": "  yellow  "}]}
    assert gate1._score_question_from_manifest(episode, q, manifest) == 1.0


# ======================================================================
# Metric construction: per-type stratification, task-averaged accuracy,
# and the visibility label for multi-slot questions.
#
# These follow LongMemEval (ICLR 2025), whose reporting guidance is to
# "prefer task_averaged_accuracy to overall_accuracy -- it corrects for
# type imbalance", and MemDelta (arXiv 2606.29914), which faults the field
# for not reporting "whether observed differences survive a significance
# test". Pooled accuracy across buckets holding different question-type
# mixes is exactly the confound that made run 2 report a NEGATIVE gap.
# ======================================================================


def _q(kind, id, supporting):
    return Question(id=id, kind=kind, text="?", answer="x",
                    supporting_fact_ids=tuple(supporting))


def test_visibility_all_slots_visible_is_visible():
    episode, q = _multi_hop_episode()
    assert gate1._question_visibility(episode, q, {"f0", "f1"}) == "visible"


def test_visibility_no_slots_visible_is_compacted():
    episode, q = _multi_hop_episode()
    assert gate1._question_visibility(episode, q, set()) == "compacted"


def test_visibility_partial_is_mixed_not_visible():
    """The bug that produced the negative gap: under an any() rule a
    multi-hop question with ONE visible slot was labelled "visible", but
    scoring requires BOTH slots right. That made "visible multi_hop"
    strictly harder than "compacted multi_hop" (0.000 vs 0.333) -- an
    artifact, not a result. Partial visibility is its own category and
    must stay out of the headline comparison."""
    episode, q = _multi_hop_episode()
    assert gate1._question_visibility(episode, q, {"f0"}) == "mixed"
    assert gate1._question_visibility(episode, q, {"f1"}) == "mixed"


def test_visibility_single_slot_question_is_never_mixed():
    f = _fact("f0", "venue_deposit", "venue deposit", "$450")
    q = _q("single_hop", "q-single-venue_deposit", ["f0"])
    episode = _episode([f], [q])
    assert gate1._question_visibility(episode, q, {"f0"}) == "visible"
    assert gate1._question_visibility(episode, q, set()) == "compacted"


def test_task_averaged_accuracy_corrects_for_type_imbalance():
    """The headline fix. Pooled accuracy is dominated by whichever type
    happens to be most numerous; task-averaged weights each type equally."""
    by_kind = {
        "single_hop": [1.0] * 16,       # many easy items
        "temporal_update": [0.0] * 2,   # few hard items
    }
    pooled = sum(sum(v) for v in by_kind.values()) / sum(len(v) for v in by_kind.values())
    assert abs(pooled - 16 / 18) < 1e-9
    assert gate1._task_averaged(by_kind) == 0.5


def test_task_averaged_accuracy_equals_pooled_when_balanced():
    by_kind = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    assert gate1._task_averaged(by_kind) == 0.5


def test_task_averaged_ignores_empty_kinds():
    by_kind = {"a": [1.0], "b": []}
    assert gate1._task_averaged(by_kind) == 1.0


def test_task_averaged_of_nothing_is_nan_not_zero():
    """An empty bucket must not read as 0.0 accuracy -- that would be a
    silent FAIL verdict from having no data, which is not the same claim."""
    import math
    assert math.isnan(gate1._task_averaged({}))
    assert math.isnan(gate1._task_averaged({"a": []}))


def test_permutation_test_detects_a_real_difference():
    p = gate1._permutation_test([1.0] * 20, [0.0] * 20, n_iter=2000, seed=0)
    assert p < 0.01


def test_permutation_test_reports_no_difference_for_identical_samples():
    p = gate1._permutation_test([0.5] * 20, [0.5] * 20, n_iter=2000, seed=0)
    assert p > 0.9


def test_permutation_test_is_underpowered_at_tiny_n():
    """n=4 vs n=16 -- the actual run-2 single_hop sample. Even a large
    apparent difference should not clear significance here, which is the
    whole reason the gate cannot return a verdict on that data."""
    p = gate1._permutation_test([1.0] * 2, [0.0] * 2, n_iter=2000, seed=0)
    assert p > 0.05


# -- verdict preconditions -------------------------------------------------
#
# The gate used to decide with a bare `else: FAIL`, which turned every
# non-PASS run into a claim about the TASK. Re-scoring the 35-episode
# gpt-oss-20b dump on 2026-08-18 showed why that is wrong: ta_vis=0.139,
# ta_comp=0.264, all three per-type permutation p-values non-significant,
# and it printed FAIL. A policy answering 14% of questions whose answers are
# still on screen tells you about the policy, not the task.

MIN = 15


def test_floor_on_visible_bucket_is_inconclusive_not_fail():
    """The exact numbers from the gpt-oss-20b re-score."""
    verdict, msg = gate1._verdict(
        ta_vis=0.139, ta_comp=0.264, n_vis=48, n_comp=52, min_per_bucket=MIN
    )
    assert verdict == "INCONCLUSIVE"
    assert "floor" in msg.lower()
    assert "FAIL" not in verdict


def test_floor_check_uses_the_threshold_implied_by_the_pass_rule():
    """PASS needs gap > 0.3 with ta_comp >= 0, so ta_vis must exceed 0.30.
    At or below that, PASS cannot be reached and FAIL is unsupportable."""
    at_threshold, _ = gate1._verdict(0.30, 0.0, 40, 40, MIN)
    assert at_threshold == "INCONCLUSIVE"
    # Just above, with a real gap, the gate is allowed to speak again.
    above, _ = gate1._verdict(0.35, 0.0, 40, 40, MIN)
    assert above == "PASS-looking"


def test_competent_policy_with_no_gap_still_fails():
    """FAIL must remain reachable -- this is the case it is FOR: the policy
    demonstrably can do the task, and compaction doesn't hurt it."""
    verdict, msg = gate1._verdict(0.80, 0.75, 40, 40, MIN)
    assert verdict == "FAIL-looking"
    assert "clears the visible bucket" in msg


def test_clear_gap_on_a_competent_policy_passes():
    verdict, _ = gate1._verdict(0.85, 0.20, 40, 40, MIN)
    assert verdict == "PASS-looking"


def test_underpowered_beats_the_floor_check():
    """Too little data is the more basic complaint: report that, not the
    floor, so the operator is told to run more episodes."""
    verdict, msg = gate1._verdict(0.05, 0.05, 3, 40, MIN)
    assert verdict == "INCONCLUSIVE"
    assert str(MIN) in msg


def test_nan_bucket_is_inconclusive_not_zero():
    verdict, msg = gate1._verdict(float("nan"), 0.2, 40, 40, MIN)
    assert verdict == "INCONCLUSIVE"
    assert "no scored questions" in msg
