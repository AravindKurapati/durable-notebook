"""Regression tests for eval/regrade_traces.py's trace replay.

Focus: the tool-call arg shapes that real training rollouts actually
produced, which the original 7-step regrade never hit because it only
sampled steps 1/10/20/30/40/50/60. Dense (every-step) regrading of the
`deployed1` runs surfaced malformed `submit_manifest` calls at
intermediate steps (e.g. stage1 step 31, 34).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from regrade_traces import reconstruct  # noqa: E402


def _rec(tool_calls: list[dict]) -> dict:
    return {"nodes": [{"message": {"role": "assistant", "tool_calls": tool_calls}}]}


def test_submit_manifest_with_bare_list_args(tmp_path: Path) -> None:
    """The policy frequently emitted `submit_manifest` with a bare JSON
    array as the whole argument payload instead of {"entries": [...]}.
    tools.py's own docstring documents this ("the model naturally wanted
    to pass the entries as a plain list"). Replay must treat the bare list
    as the entries, not crash on `list.get`."""
    entries = [
        {"slot": "venue_deposit", "path": "notes/venue.txt", "answer": "$200"},
        {"slot": "theme_color", "path": "notes/theme.txt", "answer": "emerald"},
    ]
    rec = _rec([{"name": "submit_manifest", "arguments": json.dumps(entries)}])
    _ws, manifest = reconstruct(rec, tmp_path)
    assert manifest == {"entries": entries}


def test_submit_manifest_with_entries_object_args(tmp_path: Path) -> None:
    """The well-formed shape still works."""
    entries = [{"slot": "a", "path": "p", "answer": "x"}]
    rec = _rec([{"name": "submit_manifest", "arguments": json.dumps({"entries": entries})}])
    _ws, manifest = reconstruct(rec, tmp_path)
    assert manifest == {"entries": entries}


def test_submit_manifest_with_garbage_args_is_empty(tmp_path: Path) -> None:
    """A scalar / unusable payload yields an empty manifest, matching the
    live env (no valid submission -> nothing to grade against)."""
    for payload in ('"just a string"', "42", "null"):
        rec = _rec([{"name": "submit_manifest", "arguments": payload}])
        _ws, manifest = reconstruct(rec, tmp_path)
        assert manifest == {"entries": []}


def test_write_file_escaping_path_is_skipped_not_raised(tmp_path: Path) -> None:
    """An absolute / traversing path raises WorkspaceError in the live tool,
    which catches it and returns an error string (file not written). Replay
    must not abort on it. Seen live: stage1 wrote '/home/user/notes/...'."""
    rec = _rec(
        [
            {"name": "write_file", "arguments": json.dumps({"path": "ok.txt", "content": "kept"})},
            {"name": "write_file", "arguments": json.dumps({"path": "/home/user/notes/x.txt", "content": "escapes"})},
        ]
    )
    ws, _manifest = reconstruct(rec, tmp_path)
    assert ws.read_file("ok.txt") == "kept"
    assert ws.list_files() == ["ok.txt"]


def test_write_file_non_str_content_is_skipped(tmp_path: Path) -> None:
    """A dict `content` raises TypeError inside Workspace.write_file; the
    live tool bridge caught it and returned the message to the model as a
    string (seen in real traces), so the file is not written. Replay must
    not abort."""
    rec = _rec(
        [
            {"name": "write_file", "arguments": json.dumps({"path": "ok.txt", "content": "kept"})},
            {"name": "write_file", "arguments": json.dumps({"path": "bad.txt", "content": {"a": 1}})},
        ]
    )
    ws, _manifest = reconstruct(rec, tmp_path)
    assert ws.list_files() == ["ok.txt"]


def test_write_file_replay_last_write_wins(tmp_path: Path) -> None:
    rec = _rec(
        [
            {"name": "write_file", "arguments": json.dumps({"path": "a.txt", "content": "v1"})},
            {"name": "write_file", "arguments": json.dumps({"path": "a.txt", "content": "v2"})},
        ]
    )
    ws, _manifest = reconstruct(rec, tmp_path)
    assert ws.read_file("a.txt") == "v2"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
