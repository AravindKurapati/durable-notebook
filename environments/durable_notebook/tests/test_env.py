"""Offline integration tests for DurableNotebookEnv.

These drive the environment's own methods (setup_state, get_prompt_messages,
env_response, _score_episode) directly, the same way MultiTurnEnv.rollout()
would, but with hand-authored fake model responses instead of a live model
-- no API key or network access needed. This is what's testable before
Gate 1 (which needs a real model to run the persistence ablation).
"""

import asyncio
import json

import verifiers as vf
from datasets import Dataset

from durable_notebook.env import DurableNotebookEnv
from durable_notebook.generator import generate_episode


def _row_for(episode):
    return {
        "question": episode.turns[0].text,
        "answer": "",
        "info": episode.to_dict(),
    }


def _make_env(tmp_path, grader="naive", compaction_window=3, **env_kwargs):
    episode = generate_episode(seed=1, n_turns=10, n_questions=3)
    dataset = Dataset.from_list([_row_for(episode)])
    env = DurableNotebookEnv(
        grader=grader,
        compaction_window=compaction_window,
        dataset=dataset,
        workspace_root=str(tmp_path),
        max_turns=40,
        **env_kwargs,
    )
    return env, episode


def _initial_state(env):
    row = env.get_dataset()[0]
    state = vf.State(
        info=row["info"],
        trajectory_id="test-traj",
        prompt=row["prompt"],
        trajectory=[],
    )
    return asyncio.run(env.setup_state(state)) or state


def _run(coro):
    return asyncio.run(coro)


def _assistant(content=None, tool_calls=None):
    return [vf.AssistantMessage(content=content, tool_calls=tool_calls)]


def _tool_call(name, **args):
    return vf.ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(args))


def _advance(env, state, completion):
    """One loop iteration: read the (already compacted) prompt, record a
    fake model completion, matching what MultiTurnEnv.rollout() does."""
    prompt_messages = _run(env.get_prompt_messages(state))
    state["trajectory"].append({"prompt": prompt_messages, "completion": completion})
    return prompt_messages


def _dummy_client():
    """A real verifiers Client wrapping a throwaway OpenAI client -- enough
    to satisfy resolve_client()'s isinstance check for exercising
    init_state(), without making any network call."""
    from openai import AsyncOpenAI
    from verifiers.clients import OpenAIChatCompletionsClient

    return OpenAIChatCompletionsClient(AsyncOpenAI(api_key="dummy-not-a-real-key"))


# -- construction --------------------------------------------------------


def test_env_constructs_with_expected_tools(tmp_path):
    env, _ = _make_env(tmp_path)
    assert set(env.tool_map) == {"write_file", "read_file", "list_files", "submit_manifest"}
    assert all(env.skipped_args[name] == ["state"] for name in env.tool_map)


def test_real_init_state_accepts_the_dataset_row_shape(tmp_path):
    """Drives the real Environment.init_state() (not the _initial_state()
    shortcut every other test in this file uses, which hand-builds a State
    and skips straight to setup_state). init_state() is where
    flatten_task_input()/normalize_task_payload() actually run -- a live
    Gate 1 run failed here because a dataset row had `"task": "some string"`,
    and verifiers 0.3.0 requires `task` to be a JSON object or absent, not a
    string. No test that bypasses init_state can catch that class of bug;
    this one exists specifically so the next such bug is caught for free
    instead of on a paid API call."""
    env, _ = _make_env(tmp_path)
    row = env.get_dataset()[0]
    state = _run(env.init_state(row, client=_dummy_client(), model="fake-model"))
    _run(env.setup_state(state))
    assert state["dn_episode"] is not None


def test_state_param_is_absent_from_the_schema_the_model_actually_sees(tmp_path):
    """A model can't supply a valid value for `state` -- if it leaks into
    the generated tool schema (e.g. because tools were passed straight to
    the constructor instead of via add_tool(..., args_to_skip=[...])), a
    real model would see a required argument it has no way to fill in.
    None of the tests that call tool functions directly with a hand-built
    state dict would ever catch this, since they never go through schema
    generation at all."""
    env, _ = _make_env(tmp_path)
    for tool_def in env.tool_defs:
        assert "state" not in tool_def.parameters.get("properties", {}), tool_def.name
        assert "state" not in tool_def.parameters.get("required", []), tool_def.name


def test_disable_memory_tools_leaves_only_submit_manifest(tmp_path):
    episode = generate_episode(seed=1, n_turns=10, n_questions=3)
    dataset = Dataset.from_list([_row_for(episode)])
    env = DurableNotebookEnv(
        dataset=dataset,
        workspace_root=str(tmp_path),
        disable_memory_tools=True,
    )
    assert set(env.tool_map) == {"submit_manifest"}
    assert env.system_prompt != DurableNotebookEnv(
        dataset=dataset, workspace_root=str(tmp_path / "b")
    ).system_prompt


def test_seed_prompt_is_turn_zero(tmp_path):
    env, episode = _make_env(tmp_path)
    row = env.get_dataset()[0]
    assert row["prompt"][-1]["content"] == episode.turns[0].text


# -- compaction -----------------------------------------------------------


def test_compaction_drops_old_turn_text_from_the_live_prompt(tmp_path):
    env, episode = _make_env(tmp_path, compaction_window=3)
    state = _initial_state(env)

    # Drive several plain (no tool call) turns.
    for _ in range(6):
        _advance(env, state, _assistant(content="ok"))

    final_prompt = _run(env.get_prompt_messages(state))
    final_text = json.dumps([m if isinstance(m, dict) else m.model_dump() for m in final_prompt])

    # turn 0 (seed prompt) and turn 1's text should have scrolled out of view.
    assert episode.turns[0].text not in final_text
    assert episode.turns[1].text not in final_text
    # a recent turn should still be visible.
    assert episode.turns[state["dn_turn_cursor"] - 1].text in final_text


def test_no_tools_called_override_prevents_premature_stop(tmp_path):
    env, _ = _make_env(tmp_path)
    state = _initial_state(env)
    _advance(env, state, _assistant(content="just acknowledging, no tool call"))
    assert _run(env.is_completed(state)) is False


# -- tool routing -----------------------------------------------------------


def test_write_file_tool_call_actually_writes_through_env_response(tmp_path):
    env, _ = _make_env(tmp_path)
    state = _initial_state(env)
    prompt_messages = _run(env.get_prompt_messages(state))
    tool_call = _tool_call("write_file", path="notes.txt", content="hello from the model")
    completion = _assistant(tool_calls=[tool_call])
    state["trajectory"].append({"prompt": prompt_messages, "completion": completion})

    messages_for_env_response = prompt_messages + completion
    tool_responses = _run(env.env_response(messages_for_env_response, state))

    assert state["dn_workspace"].read_file("notes.txt") == "hello from the model"
    assert any(getattr(m, "role", None) == "tool" for m in tool_responses)


# -- manifest submission ends the rollout -----------------------------------


def test_submit_manifest_sets_final_env_response_and_completes(tmp_path):
    env, _ = _make_env(tmp_path)
    state = _initial_state(env)
    entries = [{"slot": "x", "path": "x.txt", "answer": "y"}]
    state["dn_qa_shown"] = True  # submission is only legal once asked
    tool_call = _tool_call("submit_manifest", entries=entries)
    completion = _assistant(tool_calls=[tool_call])

    messages = list(state["prompt"]) + completion
    _run(env.env_response(messages, state))

    assert state["dn_manifest_submitted"] is True
    assert state["final_env_response"] is not None
    assert _run(env.is_completed(state)) is True


# -- end-to-end scoring, both graders ----------------------------------------


def _write_all_correct(env, episode, state):
    for f in episode.facts:
        _run(env.env_response(
            list(state["prompt"]) + _assistant(tool_calls=[
                _tool_call("write_file", path=f"{f.slot}.txt", content=f"{f.label}: {f.value}")
            ]),
            state,
        ))
    entries = []
    for f in episode.facts:
        entries.append({"slot": f.slot, "path": f"{f.slot}.txt", "answer": f.value})
    manifest = {"entries": entries}
    state["dn_manifest"] = manifest
    return manifest


def test_honest_full_manifest_scores_full_credit_under_both_graders(tmp_path):
    for grader in ("naive", "hardened"):
        env, episode = _make_env(tmp_path / grader, grader=grader)
        state = _initial_state(env)
        _write_all_correct(env, episode, state)
        result = _run(env._score_episode(state))
        assert result == 1.0, (grader, state["dn_grade_detail"])


def test_naive_and_hardened_diverge_on_plausible_but_wrong_content(tmp_path):
    naive_env, episode = _make_env(tmp_path / "naive", grader="naive")
    hardened_env, _ = _make_env(tmp_path / "hardened", grader="hardened")

    # Must be a fact some question actually depends on, or corrupting it
    # is invisible to both graders and the test proves nothing.
    fact_by_id = {f.id: f for f in episode.facts}
    target_fact = fact_by_id[episode.questions[0].supporting_fact_ids[0]]
    manifest_entries = [
        {"slot": f.slot, "path": f"{f.slot}.txt", "answer": f.value} for f in episode.facts
    ]

    naive_state = _initial_state(naive_env)
    hardened_state = _initial_state(hardened_env)
    for env, state in ((naive_env, naive_state), (hardened_env, hardened_state)):
        for f in episode.facts:
            content = (
                f"{f.label}: still confirming, not final yet"
                if f is target_fact
                else f"{f.label}: {f.value}"
            )
            _run(env.env_response(
                list(state["prompt"]) + _assistant(tool_calls=[
                    _tool_call("write_file", path=f"{f.slot}.txt", content=content)
                ]),
                state,
            ))
        state["dn_manifest"] = {"entries": manifest_entries}

    naive_score = _run(naive_env._score_episode(naive_state))
    hardened_score = _run(hardened_env._score_episode(hardened_state))

    assert naive_score == 1.0  # fooled by keyword-plausible content
    assert hardened_score < naive_score  # catches the wrong value


# -- compaction must not orphan tool responses ---------------------------
#
# `dn_turn_starts[i]` is the index just AFTER the model's completion for
# turn i, so every turn boundary sits at the START of the env's reply --
# which begins with the tool responses answering the assistant tool_call
# that immediately precedes it. Slicing the window at that boundary
# therefore keeps the tool messages while cutting away the assistant
# message that owns them.
#
# Some providers tolerate the resulting dangling tool messages. gpt-oss
# does not: on a live run (2026-08-17) it returned
#   400 ... HarmonyError: EncodingError: render failed: Tools should have a name!
# for 4 of 6 rollouts. It only surfaced once episodes started running to
# completion (after premature submit_manifest was blocked), because short
# rollouts never slid the window far enough to slice mid-conversation.


def _content(m):
    return m.get("content") if isinstance(m, dict) else getattr(m, "content", None)


def _roles(messages):
    return [
        m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        for m in messages
    ]


def _tool_call_ids_in_assistants(messages):
    ids = set()
    for m in messages:
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        for tc in tcs or []:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id:
                ids.add(tc_id)
    return ids


def _drive_until_compacted(tmp_path, window=3, steps=8):
    """Run enough write_file turns that the window has to slice."""
    env, _ = _make_env(tmp_path, compaction_window=window)
    state = _initial_state(env)
    messages = _run(env.get_prompt_messages(state))
    for i in range(steps):
        completion = _assistant(tool_calls=[_tool_call("write_file", path=f"f{i}.txt", content="x")])
        state["trajectory"].append({"completion": completion})
        messages = _run(env.get_prompt_messages(state))
    return messages


def test_compaction_never_leaves_a_tool_message_without_its_assistant(tmp_path):
    messages = _drive_until_compacted(tmp_path)
    known = _tool_call_ids_in_assistants(messages)
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "tool":
            continue
        tc_id = m.get("tool_call_id") if isinstance(m, dict) else getattr(m, "tool_call_id", None)
        assert tc_id in known, (
            f"tool message {tc_id!r} kept without the assistant tool_call that "
            f"produced it; roles={_roles(messages)}"
        )


def test_compaction_window_never_starts_with_a_tool_message(tmp_path):
    """The specific shape gpt-oss rejects: the visible context opening on a
    tool response (after the system/seed message)."""
    messages = _drive_until_compacted(tmp_path)
    roles = _roles(messages)
    assert roles[1] != "tool", f"window opens on a dangling tool message; roles={roles}"


def test_compaction_still_actually_compacts(tmp_path):
    """The orphan guard must not silently disable compaction -- if it kept
    everything, the whole premise of the environment would be gone."""
    messages = _drive_until_compacted(tmp_path, window=3, steps=8)
    assert len(messages) < 8 * 3, f"window not compacting; {len(messages)} messages kept"


def test_qa_intro_survives_the_model_checking_its_own_work(tmp_path):
    """Regression: the compaction window must consume budget for genuine
    *scripted* turns (fact deliveries, the QA intro) only -- not for every
    raw model step. A model that legitimately calls read_file/list_files a
    few times to double-check before answering must not thereby scroll its
    own QA instructions (and the reminder to call submit_manifest) out of
    view. Before the fix this failed with a live model: it wrote every fact
    correctly, read two files back to verify them, then -- having lost the
    QA prompt to compaction along the way -- fell into a generic
    "ready to take notes" loop for the rest of its turn budget and never
    called submit_manifest at all, scoring 0 for reasons unrelated to
    whether it tracked the facts correctly."""
    env, episode = _make_env(tmp_path, compaction_window=3)
    state = _initial_state(env)

    # Drive through every scripted fact-delivery turn with a plain
    # acknowledgement, until the QA intro has been shown.
    while not state.get("dn_qa_shown"):
        _advance(env, state, _assistant(content="ok"))
    assert state["dn_qa_shown"] is True

    # Now simulate the model checking its own work before answering:
    # several tool-call round trips, well past the compaction window size,
    # with no submit_manifest call yet.
    for i in range(6):
        _advance(env, state, _assistant(tool_calls=[_tool_call("list_files")]))

    final_prompt = _run(env.get_prompt_messages(state))
    # Compare against raw message content directly -- json.dumps would
    # escape the embedded newlines in questions_text to literal `\n` pairs,
    # which could never match unescaped content and would make this
    # assertion fail regardless of whether compaction actually kept the
    # QA intro visible.
    final_text = "\n".join(str(_content(m) or "") for m in final_prompt)
    questions_text = "\n".join(f"- {q.text}" for q in episode.questions)
    assert questions_text in final_text, (
        "QA intro (the questions + submit_manifest reminder) was compacted "
        "away by the model's own post-QA tool calls"
    )
