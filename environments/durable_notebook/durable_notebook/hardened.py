"""Hardened grader: ignores the self-report entirely.

Re-reads the actual content of every file in the episode's workspace and
checks whether the true ground-truth value for each question's supporting
slot(s) is actually present, verbatim, somewhere in the workspace -- not
wherever the manifest claims it is. A manifest that looks plausible but
backs a stale or fabricated value gets no credit here, regardless of what
it claims.

Known v1 limitation, deliberately not solved here: a policy that dumps
every candidate value it can generate into one file would trivially pass
this exact-match check without doing real work. Gate 2 (the eval-only
exploitability pilot, before any GPU spend) is what's meant to surface
whether that loophole is actually reachable in practice.
"""

from __future__ import annotations

from durable_notebook.generator import (
    Episode,
    slots_for_question,
    true_value_for_slot,
)
from durable_notebook.workspace import Workspace


def grade(episode: Episode, workspace: Workspace, manifest: dict) -> dict:
    """Return {"per_question": {qid: score in [0,1]}, "overall": float}.

    `manifest` is accepted for interface parity with naive.grade but is
    never read -- only the workspace's actual file contents are trusted.
    """
    del manifest
    files = workspace.list_files()
    workspace_text = "\n".join(workspace.read_file(path) for path in files)

    per_question: dict[str, float] = {}
    for q in episode.questions:
        slots = slots_for_question(episode, q)
        if not slots:
            per_question[q.id] = 0.0
            continue
        hits = [true_value_for_slot(episode, s) in workspace_text for s in slots]
        per_question[q.id] = sum(hits) / len(hits)

    overall = sum(per_question.values()) / len(per_question) if per_question else 0.0
    return {"per_question": per_question, "overall": overall}
