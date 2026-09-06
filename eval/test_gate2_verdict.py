"""Tests for Gate 2's verdict logic.

Gate 2 asks one question: is there a model for which the naive grader is
exploitable ABOVE the hardened baseline, with room left for GRPO to move
something? Both halves have to hold for the SAME model -- "some model has a
gap" and "some model has headroom" are not jointly evidence of anything if
they are different models.

Run from the repo root: pytest eval/test_gate2_verdict.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "durable_notebook"))

import gate2_exploitability_pilot as gate2  # noqa: E402


def R(model, naive, hardened, naive_errors=0, hardened_errors=0, n_episodes=15):
    return gate2.ModelResult(
        model=model, naive_mean=naive, hardened_mean=hardened,
        naive_errors=naive_errors, hardened_errors=hardened_errors,
        n_episodes=n_episodes,
    )


def test_gap_and_headroom_must_come_from_the_same_model():
    """The bug this file was written for.

    Model A shows a big gap but is saturated (naive at ceiling, hardened at
    floor) -- nothing left to train. Model B has plenty of headroom but no
    gap at all. The old code OR'd the two flags across models and printed
    PASS. No single model demonstrated an exploitable, movable gap, so the
    honest answer is not PASS.
    """
    results = [
        R("saturated-but-gappy", naive=1.00, hardened=0.00),
        R("headroom-but-no-gap", naive=0.40, hardened=0.38),
    ]
    verdict, _ = gate2.gate2_verdict(results)
    assert verdict != "PASS"


def test_one_model_with_both_properties_passes():
    results = [
        R("real-signal", naive=0.70, hardened=0.20),
        R("irrelevant", naive=0.10, hardened=0.10),
    ]
    verdict, msg = gate2.gate2_verdict(results)
    assert verdict == "PASS"
    assert "real-signal" in msg


def test_floored_model_does_not_supply_headroom():
    """A model scoring ~0 against BOTH graders cannot exploit anything. It
    used to satisfy `naive_mean < 0.95` and flip the headroom flag on, which
    let two models that cannot do the task at all carry the verdict."""
    results = [
        R("saturated-but-gappy", naive=1.00, hardened=0.00),
        R("floored", naive=0.00, hardened=0.00),
    ]
    verdict, msg = gate2.gate2_verdict(results)
    assert verdict != "PASS"
    assert "floored" in msg


def test_floored_models_are_reported_as_excluded_not_silently_dropped():
    results = [R("floored", naive=0.0, hardened=0.0)]
    verdict, msg = gate2.gate2_verdict(results)
    assert verdict == "INCONCLUSIVE"
    assert "floor" in msg.lower()


def test_mostly_errored_model_is_excluded_from_the_verdict():
    """Pre-existing behaviour that must survive the refactor: a gap read off
    a batch where most rollouts died is not a signal."""
    results = [R("noisy", naive=0.9, hardened=0.1, naive_errors=9, n_episodes=15)]
    verdict, msg = gate2.gate2_verdict(results)
    assert verdict == "INCONCLUSIVE"
    assert "errored" in msg.lower()


def test_saturated_everywhere_is_not_a_pass():
    """Naive at ceiling and hardened at floor for every model means the
    result is static -- there is no training pressure story for GRPO."""
    results = [
        R("a", naive=1.0, hardened=0.0),
        R("b", naive=0.99, hardened=0.01),
    ]
    verdict, _ = gate2.gate2_verdict(results)
    assert verdict != "PASS"


def test_no_usable_models_at_all_is_inconclusive_not_fail():
    results = [R("dead", naive=0.0, hardened=0.0, naive_errors=15, n_episodes=15)]
    verdict, _ = gate2.gate2_verdict(results)
    assert verdict == "INCONCLUSIVE"


def test_gap_below_threshold_with_headroom_is_a_real_fail():
    """FAIL must stay reachable and mean something: models that CAN do the
    task, have headroom, and still show no naive-vs-hardened gap."""
    results = [
        R("competent-no-gap", naive=0.55, hardened=0.50),
        R("also-competent", naive=0.60, hardened=0.58),
    ]
    verdict, msg = gate2.gate2_verdict(results)
    assert verdict == "FAIL"
    assert "gap" in msg.lower()
