"""Tool-call interface exposed to the policy: filesystem only, nothing else.

Each function takes an explicit `state` parameter carrying the per-rollout
Workspace (`state["dn_workspace"]`). The public schema shown to the model
never includes `state` -- it is injected per call by the environment's
`update_tool_args` (see env.py), which is how a single set of tool
functions can serve many concurrent rollouts without leaking one episode's
files into another's.

Errors are returned as strings rather than raised -- an unhandled exception
would otherwise abort the whole rollout instead of letting the policy see
the failure and try again, which is what an agent actually gets from a real
tool call.
"""

from __future__ import annotations

from durable_notebook.manifest import validate_manifest_shape
from durable_notebook.workspace import Workspace, WorkspaceError


def write_file(path: str, content: str, state) -> str:
    """Write `content` to `path` in your private workspace, creating parent directories as needed. Overwrites any existing file at that path."""
    workspace: Workspace = state["dn_workspace"]
    try:
        return workspace.write_file(path, content)
    except WorkspaceError as e:
        return f"error: {e}"


def read_file(path: str, state) -> str:
    """Read and return the full text content of `path` from your private workspace."""
    workspace: Workspace = state["dn_workspace"]
    try:
        return workspace.read_file(path)
    except WorkspaceError as e:
        return f"error: {e}"


def list_files(state) -> str:
    """List every file path currently in your private workspace, one per line."""
    workspace: Workspace = state["dn_workspace"]
    files = workspace.list_files()
    return "\n".join(files) if files else "(workspace is empty)"


def submit_manifest(entries: list, state) -> str:
    """Submit your final answers: a list of entries, one per question, each
    an object with "slot", "path", and "answer" fields naming the file that
    backs each answer. This ends the conversation, so only call it once you
    are done.

    A first version of this tool took the manifest as a JSON-encoded
    string parameter instead of `entries` directly. A live run against a
    real model showed it naturally wanted to pass the entries as a plain
    list -- the tool-call schema rejected that shape and the rollout
    errored out every time. Structured params matching what a model
    actually produces, not a string it has to double-encode, avoids that.
    """
    manifest = {"entries": entries}
    errors = validate_manifest_shape(manifest)
    if errors:
        return "error: " + "; ".join(errors) + ". Try again."
    # Refuse to end the episode before the questions have been asked.
    #
    # This call is what terminates the rollout, and it used to do so
    # whenever it was made -- including on turn 1, before any facts had
    # been delivered and before the QA turn existed. On a real 35-episode
    # gpt-oss-20b run (2026-08-17), 18 of 30 scored episodes never got past
    # turn 6 of 12 and 5 referenced no facts at all, because the policy
    # simply submitted and quit. Gate 1 was then measuring how far the
    # policy bothered to go, not whether the task requires persistence.
    #
    # It matters more for training than for the gate: ending the episode
    # early is a degenerate strategy GRPO is free to discover, and it skips
    # the entire task. The environment must enforce the contract rather
    # than trust the policy to play along.
    if not state.get("dn_qa_shown"):
        return (
            "error: the questions have not been asked yet, so there is "
            "nothing to answer. Keep reading the incoming turns (and "
            "recording anything you will need) until you are asked the "
            "questions, then call this once."
        )
    state["dn_manifest"] = manifest
    state["dn_manifest_submitted"] = True
    return "manifest received, thank you."


TOOLS = [write_file, read_file, list_files, submit_manifest]
STATE_ARG_NAME = "state"
