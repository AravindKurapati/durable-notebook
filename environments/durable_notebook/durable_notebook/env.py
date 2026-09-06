"""verifiers Environment wiring for durable-notebook.

Each rollout gets its own on-disk Workspace and its own Episode (decoded
from the dataset row's "info" column). The episode's scenario turns are
delivered one at a time as env-authored user messages; the model may call
the filesystem tools between turns or just acknowledge in plain text.
Compaction is enforced in `get_prompt_messages`: the prompt handed to the
model is rebuilt from scratch every step from only the last
`compaction_window` turns, plus the system message. Older turns are not
summarized, they are simply absent -- if their content matters later, it
had to be written to a file while it was still visible.

Swap the grader with `load_environment(grader="naive" | "hardened")`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import verifiers as vf

from durable_notebook import hardened as hardened_grader
from durable_notebook import naive as naive_grader
from durable_notebook.generator import Episode, episode_from_dict, generate_dataset
from durable_notebook.tools import STATE_ARG_NAME, TOOLS, submit_manifest
from durable_notebook.workspace import Workspace

GRADERS = {"naive": naive_grader, "hardened": hardened_grader}

SYSTEM_PROMPT = (
    "You are keeping notes for someone across a long conversation. Facts "
    "will be stated over many turns. Older turns will eventually stop "
    "being shown to you once they scroll out of view, so if you want to "
    "remember something you must write it to a file using the tools "
    "available to you -- there is no other way to keep it. When an update "
    "arrives for something you already noted, overwrite your notes rather "
    "than leaving the old value in place. At the end you will be asked a "
    "set of questions and must call submit_manifest with one entries list "
    'entry per question, each of the form {"slot": "...", "path": "...", '
    '"answer": "..."}, naming the file that backs each answer.'
)

ABLATION_SYSTEM_PROMPT = (
    "You are being asked questions about a long conversation. Facts will be "
    "stated over many turns. Older turns will eventually stop being shown "
    "to you once they scroll out of view -- you have no tool to write "
    "anything down, so once a turn scrolls out of view its content is "
    "genuinely gone for you. At the end you will be asked a set of "
    "questions and must call submit_manifest with one entries list entry "
    'per question, each of the form {"slot": "...", "path": "n/a", '
    '"answer": "..."}, giving your best answer from memory. Use \'n/a\' '
    "for path since you have no files."
)

QA_INTRO = (
    "That's everything I have. Now answer these questions, then call "
    "submit_manifest with your manifest:\n\n"
)


def _role_of(message) -> str | None:
    """Role of a log entry, whether it is a typed vf message or a plain
    dict -- the log carries both depending on which path produced it."""
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


class DurableNotebookEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        grader: str = "naive",
        compaction_window: int = 5,
        workspace_root: str | None = None,
        max_turns: int = 40,
        disable_memory_tools: bool = False,
        **kwargs,
    ):
        if grader not in GRADERS:
            raise ValueError(f"unknown grader {grader!r}, expected one of {list(GRADERS)}")
        self.grader_name = grader
        self.grader = GRADERS[grader]
        self.compaction_window = compaction_window
        self.workspace_root = Path(workspace_root or tempfile.mkdtemp(prefix="durable-notebook-"))
        self.disable_memory_tools = disable_memory_tools

        rubric = kwargs.pop("rubric", None) or vf.Rubric(
            funcs=[self._score_episode], weights=[1.0]
        )
        system_prompt = kwargs.pop("system_prompt", None) or (
            ABLATION_SYSTEM_PROMPT if disable_memory_tools else SYSTEM_PROMPT
        )

        super().__init__(
            tools=[],
            max_turns=max_turns,
            rubric=rubric,
            system_prompt=system_prompt,
            **kwargs,
        )
        # `state` must be filtered out of the schema the model sees (it has
        # no way to fill in a valid value for it), not just injected at call
        # time -- add_tool(..., args_to_skip=[...]) is what actually removes
        # it from the generated JSON schema. Passing tools=[...] straight to
        # the constructor above would build schemas from the raw, unfiltered
        # signatures instead.
        # For Gate 1 (persistence-necessity ablation), only submit_manifest
        # is registered -- the model has no filesystem tool at all, so
        # anything compacted out of view is genuinely gone for it, the same
        # way it would be gone even with the tools present if it never
        # wrote it down.
        active_tools = [submit_manifest] if disable_memory_tools else TOOLS
        for tool in active_tools:
            self.add_tool(tool, args_to_skip=[STATE_ARG_NAME])

    # -- tool wiring -----------------------------------------------------

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        return {**tool_args, STATE_ARG_NAME: state}

    async def no_tools_called(self, state: vf.State) -> bool:
        """Overrides ToolEnv's @vf.stop condition of the same name (not
        re-decorated, so it is dropped from discovery). A durable-notebook
        turn is allowed to be a plain acknowledgement with no tool call;
        scripted-turn delivery, not tool-call presence, drives the end of
        the rollout."""
        return False

    # -- per-rollout setup -------------------------------------------------

    async def setup_state(self, state: vf.State) -> vf.State:
        info = state.get("info") or {}
        episode = episode_from_dict(dict(info))
        workspace = Workspace(self.workspace_root / state["trajectory_id"])
        state["dn_episode"] = episode
        state["dn_workspace"] = workspace
        state["dn_turn_cursor"] = 1  # turn 0 is delivered as the seed prompt
        state["dn_qa_shown"] = False
        state["dn_manifest"] = None
        state["dn_manifest_submitted"] = False
        return state

    # -- turn delivery -----------------------------------------------------

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        last_msg = messages[-1]
        if getattr(last_msg, "tool_calls", None):
            tool_responses = await super().env_response(messages, state, **kwargs)
        else:
            tool_responses = []

        if state.get("dn_manifest_submitted"):
            state["final_env_response"] = tool_responses
            return tool_responses

        episode = cast(Episode, state["dn_episode"])
        cursor = state["dn_turn_cursor"]

        if cursor < len(episode.turns):
            next_turn = episode.turns[cursor]
            state["dn_turn_cursor"] = cursor + 1
            return [*tool_responses, vf.UserMessage(content=next_turn.text)]

        if not state["dn_qa_shown"]:
            state["dn_qa_shown"] = True
            questions_text = "\n".join(f"- {q.text}" for q in episode.questions)
            return [*tool_responses, vf.UserMessage(content=QA_INTRO + questions_text)]

        return tool_responses

    # -- environment-enforced compaction ------------------------------------

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        """Rebuild the visible prompt from an independent running log
        (`state["dn_log"]`) rather than the base class's cumulative
        per-step prompts, so a fixed trailing window of *turns* (not raw
        messages) can be sliced cleanly. `dn_turn_starts[i]` is the index
        into `dn_log` where turn i's content begins; keeping only entries
        from `dn_turn_starts[-compaction_window]` onward is exactly
        "the last `compaction_window` turns are visible, everything
        earlier is gone" -- the same rule as durable_notebook.compaction,
        applied here to the live message log instead of Episode.turns.
        """
        trajectory = state["trajectory"]
        if not trajectory:
            state["dn_log"] = list(state["prompt"])
            state["dn_turn_starts"] = [0]
            return state["prompt"]

        log = state["dn_log"] + list(trajectory[-1]["completion"])
        turn_start = len(log)

        # Snapshot scripted-turn progress before delegating to env_response,
        # so we can tell afterward whether it actually delivered a new
        # *scripted* turn (a fact-delivery UserMessage, or the QA intro) as
        # opposed to just forwarding tool_responses with nothing new. Only
        # the former should consume a slot of the compaction window --
        # otherwise every raw model step (e.g. a round of read_file calls
        # made while double-checking an answer) falsely counts as a turn,
        # and within `compaction_window` such steps the QA intro itself can
        # scroll out of view even though nothing new was ever said.
        cursor_before = state["dn_turn_cursor"]
        qa_shown_before = state["dn_qa_shown"]

        env_resp = await self.env_response(log, state)
        from verifiers.utils.message_utils import maybe_normalize_messages

        env_resp = maybe_normalize_messages(env_resp, field_name="env_response")
        log = log + list(env_resp)

        state["dn_log"] = log

        new_scripted_turn = state["dn_turn_cursor"] > cursor_before or (
            state["dn_qa_shown"] and not qa_shown_before
        )
        if new_scripted_turn:
            # A genuine new turn boundary. Once scripted turns are exhausted
            # and the QA intro has already been shown, env_response has
            # nothing left to deliver, so this branch is never taken again
            # for the rest of the rollout -- the window simply stops
            # sliding and everything from the last real boundary (the QA
            # intro) onward, including all of the model's own post-QA tool
            # calls, keeps accumulating in the visible tail.
            state["dn_turn_starts"].append(turn_start)

        starts = state["dn_turn_starts"]
        keep_from = max(0, len(starts) - self.compaction_window)
        slice_point = starts[keep_from]
        tail = log[max(1, slice_point):]

        # Drop tool responses orphaned by the cut.
        #
        # `turn_start` is the index just AFTER the model's completion, so a
        # turn boundary always sits at the start of the env's reply -- and
        # that reply begins with the tool messages answering the assistant
        # tool_call immediately before the cut. Slicing there keeps those
        # tool messages while removing the assistant message that owns them.
        #
        # Some providers tolerate the dangling result. gpt-oss does not: a
        # live run (2026-08-17) failed 4 of 6 rollouts with
        # `400 ... HarmonyError: render failed: Tools should have a name!`.
        # It stayed hidden until episodes began running to completion,
        # because short rollouts never slid the window far enough to cut
        # mid-conversation.
        #
        # Dropping them (rather than pulling the owning assistant message
        # back in) keeps the window a strict suffix of the log: reaching
        # backwards would let content the compaction rule already discarded
        # leak back into view, which is the one thing this environment
        # cannot allow.
        first_kept = 0
        while first_kept < len(tail) and _role_of(tail[first_kept]) == "tool":
            first_kept += 1
        return [log[0]] + tail[first_kept:]

    # -- scoring -------------------------------------------------------------

    async def _score_episode(self, state: vf.State, **kwargs) -> float:
        episode = cast(Episode, state["dn_episode"])
        workspace = cast(Workspace, state["dn_workspace"])
        manifest = state.get("dn_manifest") or {"entries": []}
        result = self.grader.grade(episode, workspace, manifest)
        state["dn_grade_detail"] = result
        return result["overall"]


def _episode_to_row(episode: Episode) -> dict:
    # No "task" field: verifiers 0.3.0 requires it to be either absent or a
    # JSON object (used for its own task-routing), not a plain string --
    # confirmed against verifiers.types.normalize_task_payload. A live Gate
    # 1 dry run failed on exactly this before it was caught here.
    return {
        "question": episode.turns[0].text,
        "answer": "",
        "info": episode.to_dict(),
    }


def load_environment(
    grader: str = "naive",
    compaction_window: int = 5,
    n_episodes: int = 500,
    start_seed: int = 0,
    n_turns: int = 15,
    n_questions: int = 6,
    **kwargs,
) -> DurableNotebookEnv:
    from datasets import Dataset

    train, held_out = generate_dataset(
        n_episodes=n_episodes,
        start_seed=start_seed,
        n_turns=n_turns,
        n_questions=n_questions,
    )
    dataset = Dataset.from_list([_episode_to_row(e) for e in train])
    eval_dataset = Dataset.from_list([_episode_to_row(e) for e in held_out])

    return DurableNotebookEnv(
        grader=grader,
        compaction_window=compaction_window,
        dataset=dataset,
        eval_dataset=eval_dataset,
        **kwargs,
    )
