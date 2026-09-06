import pytest

from durable_notebook.workspace import Workspace, WorkspaceError


def test_write_then_read_roundtrip(tmp_path):
    ws = Workspace(tmp_path / "episode-1")
    ws.write_file("notes/venue.txt", "venue deposit: $450")
    assert ws.read_file("notes/venue.txt") == "venue deposit: $450"


def test_list_files_reports_relative_posix_paths(tmp_path):
    ws = Workspace(tmp_path / "episode-2")
    ws.write_file("a.txt", "1")
    ws.write_file("sub/b.txt", "2")
    assert ws.list_files() == ["a.txt", "sub/b.txt"]


def test_read_missing_file_raises(tmp_path):
    ws = Workspace(tmp_path / "episode-3")
    with pytest.raises(WorkspaceError):
        ws.read_file("nope.txt")


@pytest.mark.parametrize("bad_path", [
    "../escape.txt",
    "../../etc/passwd",
    "/etc/passwd",
    "sub/../../escape.txt",
])
def test_path_traversal_is_blocked(tmp_path, bad_path):
    ws = Workspace(tmp_path / "episode-4")
    with pytest.raises(WorkspaceError):
        ws.write_file(bad_path, "x")
    with pytest.raises(WorkspaceError):
        ws.read_file(bad_path)


def test_empty_path_is_rejected(tmp_path):
    ws = Workspace(tmp_path / "episode-5")
    with pytest.raises(WorkspaceError):
        ws.write_file("", "x")


def test_two_workspaces_are_isolated(tmp_path):
    a = Workspace(tmp_path / "episode-a")
    b = Workspace(tmp_path / "episode-b")
    a.write_file("f.txt", "from a")
    with pytest.raises(WorkspaceError):
        b.read_file("f.txt")
    assert b.list_files() == []
