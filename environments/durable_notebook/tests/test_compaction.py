import pytest

from durable_notebook.compaction import compacted_turn_indices, visible_turns
from durable_notebook.generator import generate_episode


def test_window_never_exceeds_requested_size():
    episode = generate_episode(seed=1, n_turns=15)
    for i in range(len(episode.turns)):
        visible = visible_turns(list(episode.turns), i, window=5)
        assert len(visible) <= 5


def test_early_turns_are_not_padded_below_window():
    episode = generate_episode(seed=1, n_turns=15)
    turns = list(episode.turns)
    assert visible_turns(turns, 0, window=5) == turns[:1]
    assert visible_turns(turns, 2, window=5) == turns[:3]


def test_window_slides_and_drops_old_turns():
    episode = generate_episode(seed=1, n_turns=15)
    turns = list(episode.turns)
    visible = visible_turns(turns, 9, window=5)
    assert [t.index for t in visible] == [5, 6, 7, 8, 9]


def test_current_turn_always_included():
    episode = generate_episode(seed=1, n_turns=15)
    turns = list(episode.turns)
    for i in range(len(turns)):
        visible = visible_turns(turns, i, window=3)
        assert visible[-1].index == i


def test_compacted_indices_are_exactly_what_visible_turns_excludes():
    episode = generate_episode(seed=1, n_turns=15)
    turns = list(episode.turns)
    current, window = 10, 4
    visible = visible_turns(turns, current, window)
    compacted = compacted_turn_indices(turns, current, window)
    visible_indices = {t.index for t in visible}
    assert set(compacted).isdisjoint(visible_indices)
    assert set(compacted) | visible_indices == set(range(current + 1))


def test_nonpositive_window_rejected():
    episode = generate_episode(seed=1, n_turns=15)
    with pytest.raises(ValueError):
        visible_turns(list(episode.turns), 0, window=0)
