"""Environment-enforced compaction.

A fixed rule drops old scenario turns from what the policy can see, on a
schedule the environment controls -- never something the model requests or
influences. That is deliberate: if the model could choose what gets
summarized away, there would be no way to be sure the "must persist to disk
before it's gone" premise actually held for every fact.
"""

from __future__ import annotations

from durable_notebook.generator import Turn


def visible_turns(turns: list[Turn], current_index: int, window: int) -> list[Turn]:
    """Return only the turns still visible at `current_index` under a fixed
    trailing window of size `window` (inclusive of the current turn).

    Turns older than the window are compacted out entirely -- not
    summarized, just gone from context. The only way their content survives
    is if it was written to the workspace while it was still visible.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    start = max(0, current_index - window + 1)
    return turns[start:current_index + 1]


def compacted_turn_indices(turns: list[Turn], current_index: int, window: int) -> list[int]:
    """Indices of turns that have already fallen out of the visible window."""
    start = max(0, current_index - window + 1)
    return [t.index for t in turns[:start]]
