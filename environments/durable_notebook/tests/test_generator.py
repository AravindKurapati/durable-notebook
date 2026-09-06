from durable_notebook.generator import (
    SCENARIOS,
    episode_from_dict,
    generate_dataset,
    generate_episode,
    slot_label,
    slots_for_question,
    true_value_for_slot,
)
from durable_notebook.manifest import entry_for_slot, validate_manifest_shape

SEEDS = range(200)


def test_determinism():
    for seed in (0, 1, 7, 42, 999):
        a = generate_episode(seed)
        b = generate_episode(seed)
        assert a.to_dict() == b.to_dict()


def test_turn_and_scenario_shape():
    for seed in (0, 3, 11, 100):
        episode = generate_episode(seed, n_turns=15, n_questions=6)
        assert len(episode.turns) == 15
        assert episode.scenario in SCENARIOS
        for turn in episode.turns:
            assert turn.kind in ("fact", "filler")
            if turn.kind == "fact":
                assert turn.fact_id is not None
            else:
                assert turn.fact_id is None


def test_split_is_deterministic_and_matches_seed_rule():
    for seed in SEEDS:
        episode = generate_episode(seed)
        expected = "held_out" if seed % 5 == 0 else "train"
        assert episode.split == expected


def test_questions_reference_real_facts():
    for seed in SEEDS:
        episode = generate_episode(seed)
        fact_ids = {f.id for f in episode.facts}
        for q in episode.questions:
            assert q.answer.strip()
            assert q.supporting_fact_ids
            for fid in q.supporting_fact_ids:
                assert fid in fact_ids


def test_temporal_update_answer_is_the_latest_value_not_the_first():
    seen_temporal = 0
    for seed in SEEDS:
        episode = generate_episode(seed)
        updated_slots = {f.slot for f in episode.facts if f.is_update}
        for q in episode.questions:
            if q.kind != "temporal_update":
                continue
            seen_temporal += 1
            slot = q.id.removeprefix("q-temporal-")
            assert slot in updated_slots
            slot_facts = [f for f in episode.facts if f.slot == slot]
            slot_facts.sort(key=lambda f: f.turn_index)
            latest = slot_facts[-1]
            earliest = slot_facts[0]
            assert q.answer == latest.value
            if latest.value != earliest.value:
                assert q.answer != earliest.value
    assert seen_temporal > 0, "no temporal_update questions generated across sample seeds"


def test_multi_hop_questions_combine_two_distinct_facts():
    seen_multi = 0
    for seed in SEEDS:
        episode = generate_episode(seed)
        for q in episode.questions:
            if q.kind != "multi_hop":
                continue
            seen_multi += 1
            assert len(q.supporting_fact_ids) == 2
            assert q.supporting_fact_ids[0] != q.supporting_fact_ids[1]
    assert seen_multi > 0, "no multi_hop questions generated across sample seeds"


def test_single_hop_slot_was_never_updated():
    for seed in SEEDS:
        episode = generate_episode(seed)
        updated_slots = {f.slot for f in episode.facts if f.is_update}
        for q in episode.questions:
            if q.kind != "single_hop":
                continue
            slot = q.id.removeprefix("q-single-")
            assert slot not in updated_slots


def test_every_slot_revisit_is_explicitly_labeled_as_an_update():
    """A slot's value must never change without an "Update:" turn -- a second
    unlabeled "fact" turn for an already-seen slot would silently overwrite
    the value with no textual signal, which is what generate_episode used to
    do before this test was added."""
    for seed in SEEDS:
        episode = generate_episode(seed)
        by_slot: dict[str, list] = {}
        for f in episode.facts:
            by_slot.setdefault(f.slot, []).append(f)
        for slot, facts in by_slot.items():
            facts.sort(key=lambda f: f.turn_index)
            assert facts[0].is_update is False, (seed, slot)
            for later in facts[1:]:
                assert later.is_update is True, (seed, slot)
            # exactly one initial + at most one update per slot
            assert len(facts) <= 2, (seed, slot, len(facts))


def test_fact_and_update_text_use_the_expected_template_wording():
    for seed in (0, 1, 2, 3, 4, 5):
        episode = generate_episode(seed)
        for turn in episode.turns:
            if turn.kind != "fact":
                continue
            fact = next(f for f in episode.facts if f.id == turn.fact_id)
            if fact.is_update:
                assert turn.text.startswith("Update:")
            else:
                assert not turn.text.startswith("Update:")


def test_dataset_split_is_disjoint_and_covers_all_seeds():
    train, held_out = generate_dataset(n_episodes=100, start_seed=0)
    assert len(train) + len(held_out) == 100
    train_ids = {e.episode_id for e in train}
    held_ids = {e.episode_id for e in held_out}
    assert train_ids.isdisjoint(held_ids)
    assert all(e.split == "train" for e in train)
    assert all(e.split == "held_out" for e in held_out)


def test_manifest_shape_validation():
    assert validate_manifest_shape({"entries": []}) == []
    good = {"entries": [{"slot": "venue_deposit", "path": "venue_deposit.txt", "answer": "$450"}]}
    assert validate_manifest_shape(good) == []
    assert validate_manifest_shape([]) != []
    assert validate_manifest_shape({}) != []
    assert validate_manifest_shape({"entries": [{"slot": "x"}]}) != []


def test_episode_dict_roundtrip():
    for seed in (0, 1, 7, 42, 999):
        original = generate_episode(seed)
        restored = episode_from_dict(original.to_dict())
        assert restored == original


def test_slots_for_question_matches_supporting_facts():
    for seed in (0, 3, 11, 100):
        episode = generate_episode(seed)
        fact_by_id = {f.id: f for f in episode.facts}
        for q in episode.questions:
            expected = []
            for fid in q.supporting_fact_ids:
                slot = fact_by_id[fid].slot
                if slot not in expected:
                    expected.append(slot)
            assert slots_for_question(episode, q) == expected


def test_slot_label_matches_the_fact_that_defines_it():
    episode = generate_episode(0)
    for f in episode.facts:
        assert slot_label(episode, f.slot) == f.label


def test_slot_label_unknown_slot_raises():
    episode = generate_episode(0)
    try:
        slot_label(episode, "not-a-real-slot")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_true_value_for_slot_is_the_latest_value():
    for seed in SEEDS:
        episode = generate_episode(seed)
        updated_slots = {f.slot for f in episode.facts if f.is_update}
        for slot in updated_slots:
            slot_facts = sorted(
                (f for f in episode.facts if f.slot == slot), key=lambda f: f.turn_index
            )
            assert true_value_for_slot(episode, slot) == slot_facts[-1].value


def test_manifest_entry_lookup():
    manifest = {"entries": [
        {"slot": "venue_deposit", "path": "a.txt", "answer": "$450"},
        {"slot": "guest_count", "path": "b.txt", "answer": "95"},
    ]}
    entry = entry_for_slot(manifest, "guest_count")
    assert entry is not None
    assert entry["answer"] == "95"
    assert entry_for_slot(manifest, "missing_slot") is None


def test_start_seed_offsets_produce_disjoint_episodes():
    """`--start-seed` has to actually move the seed window.

    Gate 1 against a TPD-capped model can only be run in chunks, and the
    chunks are pooled afterwards (eval/merge_drip.py). If two chunks
    regenerated the same episodes, pooling them would look like more data
    while adding none -- the same facts scored twice, with the duplicates
    silently inflating the per-bucket n that MIN_PER_BUCKET checks. This
    was a live bug: `start_seed` was hardcoded to 0 in the Gate 1 runner
    before the flag existed.
    """
    a_train, a_held = generate_dataset(n_episodes=10, start_seed=0)
    b_train, b_held = generate_dataset(n_episodes=10, start_seed=50)

    a_ids = {e.episode_id for e in a_train} | {e.episode_id for e in a_held}
    b_ids = {e.episode_id for e in b_train} | {e.episode_id for e in b_held}
    assert len(a_ids) == 10 and len(b_ids) == 10
    assert a_ids.isdisjoint(b_ids)

    # Distinct ids alone would pass even if the CONTENT were identical, and
    # identical content is the failure that actually costs tokens.
    a_facts = {(f.slot, f.value) for e in a_train for f in e.facts}
    b_facts = {(f.slot, f.value) for e in b_train for f in e.facts}
    assert a_facts != b_facts


def test_start_seed_is_deterministic():
    """Same offset, same episodes -- so a chunk can be re-scored offline
    (`--rescore`) or re-run after a rate-limit abort without drifting."""
    first, _ = generate_dataset(n_episodes=5, start_seed=200)
    second, _ = generate_dataset(n_episodes=5, start_seed=200)
    assert [e.to_dict() for e in first] == [e.to_dict() for e in second]


def test_start_seed_preserves_the_train_held_out_rule():
    """The split rule is seed % 5 == 0 -> held_out. An offset window must
    keep obeying it, not accidentally send a whole chunk to one side."""
    train, held_out = generate_dataset(n_episodes=20, start_seed=200)
    assert all(e.split == "train" for e in train)
    assert all(e.split == "held_out" for e in held_out)
    assert all(e.seed % 5 != 0 for e in train)
    assert all(e.seed % 5 == 0 for e in held_out)
    assert len(held_out) == 4
