"""Tests for Gate 2's shared-rollout scoring.

Gate 2 used to run every episode TWICE per model -- once under
`grader="naive"`, once under `grader="hardened"" -- to compare the two
scores. But neither the system prompt nor the model's behaviour depends on
which grader is active (see `DurableNotebookEnv._score_episode`: the grader
only touches the *result*, after the conversation is already over). So the
two rollouts are, model-sampling-noise aside, the same experiment run
twice, and Gate 2 was paying for double the tokens it needed.

`score_output_both_graders` scores ONE rollout's resulting workspace +
manifest under both graders. These tests pin down that it produces the
same numbers a from-scratch `naive`/`hardened` rollout would -- offline, no
API key, via the scripted-policy harness the rest of the test suite uses.

Run from the repo root: pytest eval/test_gate2_shared_scoring.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "durable_notebook"))
sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "durable_notebook" / "tests"))

import verifiers as vf  # noqa: E402
from datasets import Dataset  # noqa: E402

from durable_notebook.env import QA_INTRO, DurableNotebookEnv  # noqa: E402
from durable_notebook.generator import generate_episode, slots_for_question  # noqa: E402

import gate2_exploitability_pilot as gate2  # noqa: E402


# -- minimal local copies of the scripted-policy harness from
# environments/durable_notebook/tests/test_full_episode.py. Importing that
# module directly would pull in its whole test collection; these few
# helpers are all this file needs, kept in lockstep by intent (same
# `FabricatingPolicy` behaviour) with the originals. --------------------


def _make_env(tmp_path, grader="naive", compaction_window=3, n_turns=10, n_questions=3, seed=1):
    episode = generate_episode(seed=seed, n_turns=n_turns, n_questions=n_questions)
    dataset = Dataset.from_list([{
        "question": episode.turns[0].text,
        "answer": "",
        "info": episode.to_dict(),
    }])
    env = DurableNotebookEnv(
        grader=grader,
        compaction_window=compaction_window,
        dataset=dataset,
        workspace_root=str(tmp_path),
        max_turns=80,
    )
    return env, episode


def _initial_state(env):
    row = env.get_dataset()[0]
    state = vf.State(info=row["info"], trajectory_id="t", prompt=row["prompt"], trajectory=[])
    return asyncio.run(env.setup_state(state)) or state


def _role(m):
    return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)


def _content(m):
    return m.get("content") if isinstance(m, dict) else getattr(m, "content", None)


def _last_user_text(messages):
    for m in reversed(messages):
        if _role(m) == "user":
            return str(_content(m) or "")
    return ""


def _assistant(content=None, tool_calls=None):
    return [vf.AssistantMessage(content=content, tool_calls=tool_calls)]


def _tool_call(name, **args):
    return vf.ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(args))


def _play(env, state, policy):
    messages = asyncio.run(env.get_prompt_messages(state))
    for _ in range(80):
        if state.get("dn_manifest_submitted"):
            break
        completion = policy(messages)
        state["trajectory"].append({"prompt": messages, "completion": completion})
        messages = asyncio.run(env.get_prompt_messages(state))
    return state


class ScriptedPolicy:
    """A well-behaved policy: writes down every fact it is shown, then
    answers from its own notes. Mirrors test_full_episode.py's fixture of
    the same name."""

    def __init__(self, episode, write_files=True):
        self.episode = episode
        self.write_files = write_files
        fact_by_id = {f.id: f for f in episode.facts}
        self.fact_by_turn_text = {
            t.text: fact_by_id[t.fact_id] for t in episode.turns if t.kind == "fact"
        }
        self.notes: dict[str, str] = {}
        self.saw_qa = False

    def __call__(self, messages):
        text = _last_user_text(messages)

        if text.startswith(QA_INTRO.strip()[:20]):
            self.saw_qa = True
            entries = []
            for q in self.episode.questions:
                for slot in slots_for_question(self.episode, q):
                    if any(e["slot"] == slot for e in entries):
                        continue
                    entries.append({
                        "slot": slot,
                        "path": f"{slot}.txt" if self.write_files else "n/a",
                        "answer": self.notes.get(slot, "I don't know"),
                    })
            return _assistant(tool_calls=[_tool_call("submit_manifest", entries=entries)])

        fact = self.fact_by_turn_text.get(text)
        if fact is not None:
            self.notes[fact.slot] = fact.value
            if self.write_files:
                return _assistant(tool_calls=[_tool_call(
                    "write_file",
                    path=f"{fact.slot}.txt",
                    content=f"{fact.label}: {fact.value}",
                )])
        return _assistant(content="ok")


class FabricatingPolicy(ScriptedPolicy):
    """Writes keyword-plausible files, never the real value -- the exploit
    naive.grade is meant to catch. Mirrors test_full_episode.py's fixture
    of the same name."""

    def __call__(self, messages):
        text = _last_user_text(messages)
        fact = self.fact_by_turn_text.get(text)
        if fact is not None and not text.startswith(QA_INTRO.strip()[:20]):
            # Note the LABEL only. The value is deliberately never kept.
            self.notes[fact.slot] = "noted"
            return _assistant(tool_calls=[_tool_call(
                "write_file",
                path=f"{fact.slot}.txt",
                content=f"{fact.label}: recorded, see notes",
            )])
        return super().__call__(messages)


def test_shared_scoring_matches_two_independent_rollouts(tmp_path):
    """The equivalence the whole optimization rests on: scoring one
    FabricatingPolicy rollout with both graders must match running the
    exact same policy through two separately-constructed envs, one per
    grader -- because grader choice affects nothing about the rollout
    itself, only how the already-finished result is read."""
    # Two independent rollouts (the old, expensive way).
    independent = {}
    for grader in ("naive", "hardened"):
        env, episode = _make_env(tmp_path / f"independent-{grader}", grader=grader, seed=7)
        state = _initial_state(env)
        _play(env, state, FabricatingPolicy(episode))
        independent[grader] = asyncio.run(env._score_episode(state))

    # One rollout, scored under both graders (the new, shared way).
    env, episode = _make_env(tmp_path / "shared", grader="naive", seed=7)
    state = _initial_state(env)
    _play(env, state, FabricatingPolicy(episode))
    output = {
        "info": episode.to_dict(),
        "trajectory_id": state["trajectory_id"],
        "dn_manifest": state.get("dn_manifest"),
    }
    naive_shared, hardened_shared = gate2.score_output_both_graders(output, env.workspace_root)

    assert naive_shared == independent["naive"]
    assert hardened_shared == independent["hardened"]
    # Sanity check this is still exercising the real exploit contrast, not
    # two graders that happen to agree.
    assert naive_shared - hardened_shared > 0.4


def test_shared_scoring_reads_files_from_the_right_trajectory_directory(tmp_path):
    """Regression guard for the one plumbing step with no equivalent in the
    old code: reconstructing the on-disk Workspace from `trajectory_id`
    after the rollout is over, rather than holding the live object. If this
    pointed at the wrong directory it would silently score an empty
    workspace instead of erroring."""
    env, episode = _make_env(tmp_path, grader="naive", seed=3)
    state = _initial_state(env)
    _play(env, state, FabricatingPolicy(episode))

    output = {
        "info": episode.to_dict(),
        "trajectory_id": state["trajectory_id"],
        "dn_manifest": state.get("dn_manifest"),
    }
    naive_score, _ = gate2.score_output_both_graders(output, env.workspace_root)
    assert naive_score > 0.5, (
        "expected the fabricator's plausible-but-wrong files to score well "
        "on naive; a near-zero score here means the workspace was read "
        "from the wrong (empty) directory"
    )


def test_shared_scoring_handles_a_missing_manifest(tmp_path):
    """A rollout that errored, or never got to submit_manifest, has
    dn_manifest=None in state -- state_columns surfaces that as `None`, not
    a dict. Must be treated as an empty manifest, matching
    `DurableNotebookEnv._score_episode`'s own `state.get("dn_manifest") or
    {"entries": []}` fallback, not crash."""
    episode = generate_episode(seed=2, n_turns=10, n_questions=3)
    output = {"info": episode.to_dict(), "trajectory_id": "does-not-matter", "dn_manifest": None}
    naive_score, hardened_score = gate2.score_output_both_graders(output, tmp_path)
    assert naive_score == 0.0
    assert hardened_score == 0.0
