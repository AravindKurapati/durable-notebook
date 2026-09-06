"""A filesystem-only sandbox scoped to one episode's workspace directory.

No code execution ever happens here. This is the entire surface a policy
has for persisting information across the environment-enforced compaction
boundary: write it to a file, or lose it when the turn is dropped.
"""

from __future__ import annotations

from pathlib import Path


class WorkspaceError(Exception):
    """Raised for disallowed paths or other sandbox violations."""


class Workspace:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel_path: str) -> Path:
        if not rel_path or not isinstance(rel_path, str):
            raise WorkspaceError(f"invalid path: {rel_path!r}")
        parts = Path(rel_path).parts
        if Path(rel_path).is_absolute() or ".." in parts:
            raise WorkspaceError(f"path escapes workspace: {rel_path!r}")
        target = (self.root / rel_path).resolve()
        if target != self.root and self.root not in target.parents:
            raise WorkspaceError(f"path escapes workspace: {rel_path!r}")
        return target

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"no such file: {path!r}")
        return target.read_text(encoding="utf-8")

    def list_files(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in self.root.rglob("*")
            if p.is_file()
        )
