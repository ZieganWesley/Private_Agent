"""Workspace-bounded file and directory tools."""

import contextlib
import difflib
import fnmatch
import hashlib
import logging
import os
import re
import selectors
import stat
import subprocess
import time
import uuid
from pathlib import Path

from langchain_core.tools import tool

from code_tasks.snapshots import SnapshotError, record_snapshot
from config import (
    BINARY_PROBE_BYTES,
    FILE_READ_CHUNK_BYTES,
    MAX_READ_FILE_BYTES,
)
from sandbox import SandboxManager

from .schemas import (
    ApplyWorkspacePatchInput,
    CommitGitChangesInput,
    CreateGitBranchInput,
    CreatePullRequestInput,
    DeleteWorkspaceFileInput,
    EditFileInput,
    InspectWorkspaceGitInput,
    ListDirInput,
    PreviewWorkspacePatchInput,
    ReadFileInput,
    RenameWorkspaceFileInput,
    SearchWorkspaceFilesInput,
    SearchWorkspaceSymbolsInput,
    SearchWorkspaceTextInput,
    StageGitChangesInput,
)

_LOGGER = logging.getLogger(__name__)

_MAX_SEARCH_FILES = 10_000
_MAX_SEARCH_BYTES = 16 * 1024 * 1024
_MAX_SEARCH_OUTPUT_CHARS = 12_000
_GIT_INSPECTION_TIMEOUT_SECONDS = 5
_SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "build",
    "dist",
    "target",
}
_SYMBOL_DECLARATIONS = {
    ".py": re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+(?P<name>[A-Za-z_]\w*)\b"),
    ".js": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function|class)\s+(?P<name>[A-Za-z_$][\w$]*)\b"
    ),
    ".jsx": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function|class)\s+(?P<name>[A-Za-z_$][\w$]*)\b"
    ),
    ".ts": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function|class|interface|type|enum)\s+(?P<name>[A-Za-z_$][\w$]*)\b"
    ),
    ".tsx": re.compile(
        r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"(?:function|class|interface|type|enum)\s+(?P<name>[A-Za-z_$][\w$]*)\b"
    ),
    ".go": re.compile(
        r"^\s*(?:func\s+(?:\([^)]*\)\s*)?|type\s+)"
        r"(?P<name>[A-Za-z_]\w*)\b"
    ),
    ".rs": re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
        r"(?:fn|struct|enum|trait|type|mod)\s+(?P<name>[A-Za-z_]\w*)\b"
    ),
}


def _run_git_bounded(command: list[str], environment: dict[str, str]):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    output = bytearray()
    deadline = time.monotonic() + _GIT_INSPECTION_TIMEOUT_SECONDS
    truncated = False
    timed_out = False

    def terminate():
        with contextlib.suppress(ProcessLookupError):
            process.kill()

    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = process.poll() is None
                if timed_out:
                    terminate()
                break
            events = selector.select(min(remaining, 0.1))
            if not events:
                if process.poll() is not None:
                    break
                continue
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            available = _MAX_SEARCH_OUTPUT_CHARS - len(output)
            output.extend(chunk[:available])
            if len(chunk) > available:
                truncated = True
                terminate()
                break
    returncode = process.wait()
    return returncode, output.decode("utf-8", errors="replace"), truncated, timed_out


def _workspace_files(root: Path, directory: Path):
    visited = 0
    for current, directories, filenames in os.walk(directory, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if name not in _SKIP_DIRECTORIES
            and not name.startswith(".")
            and not (current_path / name).is_symlink()
        ]
        for filename in filenames:
            path = current_path / filename
            visited += 1
            if visited > _MAX_SEARCH_FILES:
                return
            if filename.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            if resolved.is_relative_to(root):
                yield path


def _validate_search_directory(directory: str) -> tuple[Path, Path]:
    root = SandboxManager.get_root().resolve()
    target = SandboxManager.validate_path(directory)
    if not target.is_dir():
        raise ValueError(f"'{directory}' is not a workspace directory.")
    return root, target


def _path_contains_symlink(path: str) -> bool:
    root = SandboxManager.get_root().resolve()
    raw_path = Path(path)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if current.is_symlink():
            return True
    return False


def _read_workspace_text(target: Path, file_path: str) -> str:
    if not target.is_file():
        raise ValueError(f"'{file_path}' is not a regular workspace file.")
    if target.stat().st_size > MAX_READ_FILE_BYTES:
        raise ValueError(
            f"File exceeds the maximum allowed size of {MAX_READ_FILE_BYTES} bytes."
        )
    content = target.read_bytes()
    if b"\x00" in content:
        raise ValueError(f"'{file_path}' appears to be binary.")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"'{file_path}' is not valid UTF-8 text.") from exc


def _exact_patch(target: Path, file_path: str, old_text: str, new_text: str):
    current = _read_workspace_text(target, file_path)
    if (
        len(old_text.encode("utf-8")) + len(new_text.encode("utf-8"))
        > MAX_READ_FILE_BYTES
    ):
        raise ValueError("Patch content exceeds the configured file size limit.")
    occurrences = current.count(old_text)
    if occurrences != 1:
        raise ValueError(
            f"Patch context must match exactly once; found {occurrences} matches."
        )
    updated = current.replace(old_text, new_text, 1)
    if len(updated.encode("utf-8")) > MAX_READ_FILE_BYTES:
        raise ValueError("Patched file would exceed the configured file size limit.")
    diff = "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        )
    )
    return current, updated, diff


def _secure_mutation_available() -> bool:
    """Whether descriptor-relative, no-follow filesystem operations are supported."""
    return (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.link in os.supports_dir_fd
        and os.rename in os.supports_dir_fd
        and hasattr(os, "pread")
    )


def _open_workspace_parent(file_path: str, *, create: bool = False):
    if not _secure_mutation_available():
        raise OSError(
            "Secure workspace mutations require descriptor-relative filesystem "
            "operations, which are unavailable on this platform."
        )
    root = SandboxManager.get_root().resolve()
    raw_path = Path(file_path)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError("Path is outside the active workspace.") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("A workspace file path is required.")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(root, directory_flags)
    try:
        for part in relative.parts[:-1]:
            try:
                child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        return parent_fd, relative.name
    except (ValueError, OSError):
        os.close(parent_fd)
        raise


def _open_workspace_file(file_path: str):
    parent_fd, name = _open_workspace_parent(file_path)
    try:
        file_fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(file_fd)
            raise ValueError("Only regular workspace files can be modified.")
        return parent_fd, name, file_fd, file_stat
    except (ValueError, OSError):
        os.close(parent_fd)
        raise


def _snapshot_file_descriptor(
    file_fd: int,
    name: str,
    *,
    max_bytes: int | None = None,
) -> tuple[str, int, str]:
    """Copy the open file into the snapshot directory and hash what was copied.

    Returns ``(backup_name, size, sha256)`` so the caller can record a manifest
    entry without reading the backup a second time.
    """
    root_fd = os.open(
        SandboxManager.get_root().resolve(),
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        with contextlib.suppress(FileExistsError):
            os.mkdir(".agent_snapshots", mode=0o700, dir_fd=root_fd)
        snapshot_fd = os.open(
            ".agent_snapshots",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        try:
            snapshot_name = f"{name}_{uuid.uuid4().hex}.bak"
            backup_fd = os.open(
                snapshot_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=snapshot_fd,
            )
            backup_fd_closed = False
            source_stat = os.fstat(file_fd)
            digest = hashlib.sha256()
            try:
                offset = 0
                while chunk := os.pread(file_fd, 1024 * 1024, offset):
                    if max_bytes is not None and offset + len(chunk) > max_bytes:
                        raise ValueError(
                            "Workspace file grew beyond the snapshot size limit."
                        )
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(backup_fd, view)
                        view = view[written:]
                    offset += len(chunk)
                os.fsync(backup_fd)
                current_stat = os.fstat(file_fd)
                if (
                    current_stat.st_size != offset
                    or source_stat.st_size != offset
                    or current_stat.st_mtime_ns != source_stat.st_mtime_ns
                    or current_stat.st_ctime_ns != source_stat.st_ctime_ns
                ):
                    raise OSError("Workspace file changed while creating its snapshot.")
            except (ValueError, OSError):
                os.close(backup_fd)
                backup_fd_closed = True
                os.unlink(snapshot_name, dir_fd=snapshot_fd)
                raise
            finally:
                if not backup_fd_closed:
                    os.close(backup_fd)
            return snapshot_name, offset, digest.hexdigest()
        finally:
            os.close(snapshot_fd)
    finally:
        os.close(root_fd)


def _record_edit_snapshot(
    file_path: str,
    backup: tuple[str, int, str] | None,
    *,
    action: str | None = None,
) -> None:
    """Record the edit for `/undo`; never fail the edit because of the manifest.

    ``backup`` is the tuple returned by :func:`_snapshot_file_descriptor` for a
    pre-existing file, or ``None`` for a path this turn created.
    """
    from . import get_tool_context

    context = get_tool_context()
    if not context.session_id or not context.turn_id:
        return
    resolved_action = action or ("modify" if backup is not None else "create")
    try:
        record_snapshot(
            root=SandboxManager.get_root(),
            target=file_path,
            backup=backup[0] if backup is not None else None,
            action=resolved_action,
            size=backup[1] if backup is not None else 0,
            sha256=backup[2] if backup is not None else "",
            session_id=context.session_id,
            turn_id=context.turn_id,
        )
    except (OSError, ValueError, SnapshotError) as exc:
        _LOGGER.warning(
            "Could not record the snapshot manifest entry for %s: %s",
            file_path,
            str(exc)[:200],
        )


def _read_file_descriptor(file_fd: int, file_stat: os.stat_result) -> str:
    if file_stat.st_size > MAX_READ_FILE_BYTES:
        raise ValueError(
            f"File exceeds the maximum allowed size of {MAX_READ_FILE_BYTES} bytes."
        )
    chunks = []
    offset = 0
    while chunk := os.pread(
        file_fd, min(FILE_READ_CHUNK_BYTES, MAX_READ_FILE_BYTES + 1 - offset), offset
    ):
        chunks.append(chunk)
        offset += len(chunk)
        if offset > MAX_READ_FILE_BYTES:
            raise ValueError(
                f"File exceeds the maximum allowed size of {MAX_READ_FILE_BYTES} bytes."
            )
    content = b"".join(chunks)
    if b"\x00" in content:
        raise ValueError("The workspace file appears to be binary.")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("The workspace file is not valid UTF-8 text.") from exc


def _write_workspace_replacement(
    parent_fd: int,
    name: str,
    expected_stat: os.stat_result | None,
    content: bytes,
) -> None:
    file_mode = (
        stat.S_IMODE(expected_stat.st_mode) if expected_stat is not None else 0o600
    )
    temporary_name = f".{name}.{uuid.uuid4().hex}.tmp"
    temporary_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        file_mode,
        dir_fd=parent_fd,
    )
    try:
        view = memoryview(content)
        while view:
            written = os.write(temporary_fd, view)
            view = view[written:]
        os.fsync(temporary_fd)
    except (ValueError, OSError):
        os.close(temporary_fd)
        os.unlink(temporary_name, dir_fd=parent_fd)
        raise
    os.close(temporary_fd)
    try:
        try:
            current_stat = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            current_stat = None
        if expected_stat is None:
            if current_stat is not None:
                raise FileExistsError(
                    "Workspace destination appeared during the write."
                )
            os.link(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=parent_fd)
        else:
            if (
                current_stat is None
                or not stat.S_ISREG(current_stat.st_mode)
                or (current_stat.st_dev, current_stat.st_ino)
                != (expected_stat.st_dev, expected_stat.st_ino)
                or current_stat.st_nlink != 1
                or current_stat.st_size != expected_stat.st_size
                or current_stat.st_mtime_ns != expected_stat.st_mtime_ns
                or current_stat.st_ctime_ns != expected_stat.st_ctime_ns
            ):
                raise OSError("Workspace file changed during the approved write.")
            os.rename(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
    except (ValueError, OSError):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_fd)
        raise


def _write_workspace_file_safely(file_path: str, content: bytes) -> None:
    parent_fd, name = _open_workspace_parent(file_path, create=True)
    file_fd = None
    try:
        try:
            original_stat = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            original_stat = None
        backup = None
        if original_stat is not None:
            if not stat.S_ISREG(original_stat.st_mode):
                raise ValueError("Only regular workspace files can be overwritten.")
            file_fd = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            opened_stat = os.fstat(file_fd)
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                original_stat.st_dev,
                original_stat.st_ino,
            ) or opened_stat.st_nlink > 1:
                raise OSError("Workspace file changed before the write.")
            backup = _snapshot_file_descriptor(file_fd, name)
        _write_workspace_replacement(parent_fd, name, original_stat, content)
        _record_edit_snapshot(file_path, backup)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _delete_workspace_file_safely(file_path: str) -> None:
    parent_fd, name, file_fd, original_stat = _open_workspace_file(file_path)
    try:
        backup = _snapshot_file_descriptor(file_fd, name)
        current_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(current_stat.st_mode) or (
            current_stat.st_dev,
            current_stat.st_ino,
        ) != (original_stat.st_dev, original_stat.st_ino):
            raise OSError("Workspace file changed during deletion.")
        os.unlink(name, dir_fd=parent_fd)
        _record_edit_snapshot(file_path, backup)
    finally:
        os.close(file_fd)
        os.close(parent_fd)


def _apply_workspace_patch_safely(
    file_path: str,
    old_text: str,
    new_text: str,
) -> str:
    parent_fd, name, file_fd, original_stat = _open_workspace_file(file_path)
    try:
        if original_stat.st_nlink > 1:
            raise ValueError("Files with multiple hard links are not modified.")
        current = _read_file_descriptor(file_fd, original_stat)
        if (
            len(old_text.encode("utf-8")) + len(new_text.encode("utf-8"))
            > MAX_READ_FILE_BYTES
        ):
            raise ValueError("Patch content exceeds the configured file size limit.")
        occurrences = current.count(old_text)
        if occurrences != 1:
            raise ValueError(
                f"Patch context must match exactly once; found {occurrences} matches."
            )
        updated = current.replace(old_text, new_text, 1)
        updated_bytes = updated.encode("utf-8")
        if len(updated_bytes) > MAX_READ_FILE_BYTES:
            raise ValueError(
                "Patched file would exceed the configured file size limit."
            )
        diff = "".join(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=file_path,
                tofile=file_path,
            )
        )
        if not diff:
            return "Patch contains no changes."
        backup = _snapshot_file_descriptor(file_fd, name, max_bytes=MAX_READ_FILE_BYTES)
        _write_workspace_replacement(parent_fd, name, original_stat, updated_bytes)
        _record_edit_snapshot(file_path, backup)
        return f"Success: Applied patch to '{file_path}'.\n{diff}"
    finally:
        os.close(file_fd)
        os.close(parent_fd)


def _rename_workspace_file_safely(source_path: str, destination_path: str) -> str:
    source_parent_fd, source_name, source_fd, source_stat = _open_workspace_file(
        source_path
    )
    destination_parent_fd = None
    try:
        if source_stat.st_nlink > 1:
            raise ValueError("Files with multiple hard links are not renamed.")
        destination_parent_fd, destination_name = _open_workspace_parent(
            destination_path,
            create=True,
        )
        try:
            os.stat(
                destination_name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Destination '{destination_path}' already exists.")

        backup = _snapshot_file_descriptor(source_fd, source_name)
        os.link(
            source_name,
            destination_name,
            src_dir_fd=source_parent_fd,
            dst_dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        try:
            linked_stat = os.stat(
                destination_name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(linked_stat.st_mode) or (
                linked_stat.st_dev,
                linked_stat.st_ino,
            ) != (source_stat.st_dev, source_stat.st_ino):
                raise OSError("Source file changed during the rename.")
            current_source_stat = os.stat(
                source_name,
                dir_fd=source_parent_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(current_source_stat.st_mode) or (
                current_source_stat.st_dev,
                current_source_stat.st_ino,
            ) != (source_stat.st_dev, source_stat.st_ino):
                raise OSError("Source file changed during the rename.")
            os.unlink(source_name, dir_fd=source_parent_fd)
        except (ValueError, OSError):
            os.unlink(destination_name, dir_fd=destination_parent_fd)
            raise
        _record_edit_snapshot(source_path, backup)
        _record_edit_snapshot(destination_path, None, action="create")
        return f"Success: Renamed '{source_path}' to '{destination_path}'."
    finally:
        if destination_parent_fd is not None:
            os.close(destination_parent_fd)
        os.close(source_fd)
        os.close(source_parent_fd)


@tool(args_schema=ReadFileInput)
def read_local_file(file_path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a text file inside the validated workspace with configured size limits."""
    try:
        target = SandboxManager.validate_path(file_path)
        if not target.exists():
            return f"Error: File '{file_path}' not found."

        max_size = MAX_READ_FILE_BYTES
        file_size = target.stat().st_size
        if file_size > max_size:
            return (
                f"Error: File '{file_path}' exceeds the maximum allowed size of "
                f"{max_size} bytes."
            )

        with target.open("rb") as source:
            header_bytes = source.read(BINARY_PROBE_BYTES)
            if b"\x00" in header_bytes:
                return (
                    f"Error: File '{file_path}' appears to be a binary file and "
                    "cannot be read as text."
                )

        chunks = []
        with target.open("rb") as source:
            while chunk := source.read(FILE_READ_CHUNK_BYTES):
                chunks.append(chunk)
        text = b"".join(chunks).decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        if offset >= total_lines and total_lines > 0:
            return (
                f"File '{file_path}' has {total_lines} lines; "
                f"offset {offset} is out of range."
            )
        end = total_lines if limit is None else min(total_lines, offset + limit)
        selected = lines[offset:end]
        body = "".join(selected)
        return (
            f"{body}\n[read_local_file: lines {offset + 1}-{end} of "
            f"{total_lines} total]"
        )
    except (ValueError, OSError) as exc:
        return f"Error reading file: {exc}"


@tool(args_schema=EditFileInput)
def edit_local_file(file_path: str, content: str) -> str:
    """Create or overwrite a workspace file after creating a snapshot backup."""
    try:
        if _path_contains_symlink(file_path):
            return "Error writing file: Symlink paths are not modified."
        SandboxManager.validate_path(file_path)
        _write_workspace_file_safely(file_path, content.encode("utf-8"))
        return f"Success: File '{file_path}' written securely (HITL checkpoint snapshot created)."
    except (ValueError, OSError) as exc:
        return f"Error writing file: {exc}"


@tool(args_schema=ListDirInput)
def list_directory(dir_path: str = ".") -> str:
    """List files and subdirectories within the validated workspace."""
    try:
        target = SandboxManager.validate_path(dir_path)
        if not target.is_dir():
            return f"Error: '{dir_path}' is not a valid directory."
        items = []
        for child in target.iterdir():
            if child.name.startswith("."):
                continue
            prefix = "[DIR]  " if child.is_dir() else "[FILE] "
            relative_path = child.relative_to(SandboxManager.get_root())
            items.append(f"{prefix} {relative_path}")
        return "\n".join(sorted(items)) if items else "Directory is empty."
    except (ValueError, OSError) as exc:
        return f"Error listing directory: {exc}"


@tool(args_schema=SearchWorkspaceFilesInput)
def search_workspace_files(
    pattern: str,
    directory: str = ".",
    limit: int = 100,
) -> str:
    """Search bounded workspace files by relative glob pattern without following symlinks."""
    try:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            return "Error: File pattern must be workspace-relative and cannot traverse parents."
        root, target = _validate_search_directory(directory)
        matches = []
        for path in _workspace_files(root, target):
            relative = path.relative_to(root).as_posix()
            if fnmatch.fnmatchcase(relative, pattern) or fnmatch.fnmatchcase(
                path.name, pattern
            ):
                matches.append(relative)
                if len(matches) >= limit:
                    break
        if not matches:
            return "No matching workspace files."
        result = "\n".join(matches)
        if len(result) > _MAX_SEARCH_OUTPUT_CHARS:
            result = result[:_MAX_SEARCH_OUTPUT_CHARS] + "\n[Search output truncated.]"
        return result
    except (ValueError, OSError) as exc:
        return f"Error searching workspace files: {exc}"


@tool(args_schema=SearchWorkspaceTextInput)
def search_workspace_text(
    query: str,
    directory: str = ".",
    pattern: str = "*",
    limit: int = 50,
) -> str:
    """Search bounded UTF-8 workspace text, skipping hidden, binary, oversized and symlinked files."""
    try:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            return "Error: File pattern must be workspace-relative and cannot traverse parents."
        root, target = _validate_search_directory(directory)
        needle = query.casefold()
        matches = []
        scanned_bytes = 0
        truncated_scan = False
        for path in _workspace_files(root, target):
            relative = path.relative_to(root).as_posix()
            if not (
                fnmatch.fnmatchcase(relative, pattern)
                or fnmatch.fnmatchcase(path.name, pattern)
            ):
                continue
            size = path.stat().st_size
            if size > MAX_READ_FILE_BYTES:
                continue
            if scanned_bytes + size > _MAX_SEARCH_BYTES:
                truncated_scan = True
                break
            scanned_bytes += size
            remaining_bytes = _MAX_SEARCH_BYTES - (scanned_bytes - size)
            read_limit = min(MAX_READ_FILE_BYTES, remaining_bytes)
            with path.open("rb") as source:
                content = source.read(read_limit + 1)
            if len(content) > read_limit:
                truncated_scan = True
                break
            if b"\x00" in content:
                continue
            try:
                lines = content.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, 1):
                if needle in line.casefold():
                    matches.append(f"{relative}:{line_number}: {line[:500]}")
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        result = "No matching workspace text." if not matches else "\n".join(matches)
        if truncated_scan:
            result += "\n[Search stopped at the configured workspace scan limit.]"
        if len(result) > _MAX_SEARCH_OUTPUT_CHARS:
            result = result[:_MAX_SEARCH_OUTPUT_CHARS] + "\n[Search output truncated.]"
        return result
    except (ValueError, OSError) as exc:
        return f"Error searching workspace text: {exc}"


@tool(args_schema=SearchWorkspaceSymbolsInput)
def search_workspace_symbols(
    symbol: str,
    directory: str = ".",
    pattern: str = "*",
    limit: int = 50,
) -> str:
    """Find exact symbol declarations in bounded Python, JS/TS, Go, and Rust source."""
    try:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            return "Error: File pattern must be workspace-relative and cannot traverse parents."
        root, target = _validate_search_directory(directory)
        expected_name = symbol.strip().casefold()
        if not expected_name:
            return "Error: Provide a symbol name to search for."
        matches = []
        scanned_bytes = 0
        truncated_scan = False
        for path in _workspace_files(root, target):
            declaration_pattern = _SYMBOL_DECLARATIONS.get(path.suffix.lower())
            if declaration_pattern is None:
                continue
            relative = path.relative_to(root).as_posix()
            if not (
                fnmatch.fnmatchcase(relative, pattern)
                or fnmatch.fnmatchcase(path.name, pattern)
            ):
                continue
            size = path.stat().st_size
            if size > MAX_READ_FILE_BYTES:
                continue
            if scanned_bytes + size > _MAX_SEARCH_BYTES:
                truncated_scan = True
                break
            with path.open("rb") as source:
                content = source.read(size + 1)
            if len(content) != size or b"\x00" in content:
                continue
            scanned_bytes += size
            try:
                lines = content.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                declaration = declaration_pattern.match(line)
                if (
                    declaration is None
                    or declaration.group("name").casefold() != expected_name
                ):
                    continue
                matches.append(f"{relative}:{line_number}: {line.strip()[:300]}")
                if len(matches) >= limit:
                    break
            if len(matches) >= limit:
                break
        if not matches:
            result = f"No declarations found for symbol '{symbol.strip()}'."
        else:
            result = "\n".join(matches)
        if truncated_scan:
            result += "\n[Search stopped at the configured workspace scan limit.]"
        if len(result) > _MAX_SEARCH_OUTPUT_CHARS:
            result = result[:_MAX_SEARCH_OUTPUT_CHARS] + "\n[Search output truncated.]"
        return result
    except (ValueError, OSError) as exc:
        return f"Error searching workspace symbols: {exc}"


@tool(args_schema=InspectWorkspaceGitInput)
def inspect_workspace_git(action: str, directory: str = ".") -> str:
    """Read Git status, diff, recent log, or branches without modifying the repository."""
    commands = {
        "status": ["status", "--short", "--branch", "--untracked-files=normal"],
        "diff": ["diff", "--no-ext-diff", "--no-textconv"],
        "log": ["log", "-10", "--oneline", "--decorate=short", "--no-show-signature"],
        "branches": ["branch", "--list", "--no-color"],
    }
    if action not in commands:
        return "Error: Unsupported Git inspection action."
    try:
        root, target = _validate_search_directory(directory)
        environment = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
        }
        repository_command = [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-C",
            str(target),
            "rev-parse",
            "--show-toplevel",
        ]
        repository_code, repository_output, _, repository_timeout = _run_git_bounded(
            repository_command,
            environment,
        )
        if repository_timeout:
            return f"Error: Git {action} exceeded the {_GIT_INSPECTION_TIMEOUT_SECONDS}-second limit."
        if repository_code != 0:
            return "No Git repository was found in the requested workspace directory."
        repository_root = Path(repository_output.strip()).resolve()
        if not repository_root.is_relative_to(root):
            return "Error: Git repository root is outside the active workspace."
        git_command = [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-C",
            str(target),
            *commands[action],
        ]
        result_code, output, truncated, timed_out = _run_git_bounded(
            git_command,
            environment,
        )
        if timed_out:
            return f"Error: Git {action} exceeded the {_GIT_INSPECTION_TIMEOUT_SECONDS}-second limit."
        if result_code != 0:
            return f"Git {action} failed: {output[:1000]}"
        output = output.strip() or f"Git {action}: no output."
        if truncated:
            output += "\n[Git output truncated at the configured limit.]"
        return output
    except (OSError, ValueError) as exc:
        return f"Error inspecting workspace Git {action}: {exc}"


def _git_write_environment() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _git_repository_root(directory: str):
    root, target = _validate_search_directory(directory)
    environment = _git_write_environment()
    repository_command = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.pager=cat",
        "-C",
        str(target),
        "rev-parse",
        "--show-toplevel",
    ]
    code, output, _, timed_out = _run_git_bounded(repository_command, environment)
    if timed_out:
        raise ValueError("Git operation timed out.")
    if code != 0:
        raise ValueError(
            "No Git repository was found in the requested workspace directory."
        )
    repository_root = Path(output.strip()).resolve()
    if not repository_root.is_relative_to(root):
        raise ValueError("Git repository root is outside the active workspace.")
    return root, target, repository_root, environment


@tool(args_schema=CreateGitBranchInput)
def create_git_branch(branch_name: str, directory: str = ".") -> str:
    """Create and check out a new Git branch in the workspace repository."""
    try:
        _, target, _, environment = _git_repository_root(directory)
        command = [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-C",
            str(target),
            "checkout",
            "-b",
            branch_name,
        ]
        code, output, _, timed_out = _run_git_bounded(command, environment)
        if timed_out:
            return "Error: Git branch creation timed out."
        if code != 0:
            return f"Git branch creation failed: {output.strip()[:1000]}"
        return f"Created and checked out branch '{branch_name}'."
    except (OSError, ValueError) as exc:
        return f"Error creating Git branch: {exc}"


@tool(args_schema=StageGitChangesInput)
def stage_git_changes(paths: list[str] | None = None, directory: str = ".") -> str:
    """Stage workspace changes for the next commit."""
    try:
        _, target, _, environment = _git_repository_root(directory)
        selected_paths = paths or ["."]
        command = [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-C",
            str(target),
            "add",
            "--",
            *selected_paths,
        ]
        code, output, _, timed_out = _run_git_bounded(command, environment)
        if timed_out:
            return "Error: Git staging timed out."
        if code != 0:
            return f"Git staging failed: {output.strip()[:1000]}"
        return f"Staged {len(selected_paths)} path(s)."
    except (OSError, ValueError) as exc:
        return f"Error staging Git changes: {exc}"


@tool(args_schema=CommitGitChangesInput)
def commit_git_changes(message: str, directory: str = ".") -> str:
    """Create a Git commit with the staged changes."""
    try:
        _, target, _, environment = _git_repository_root(directory)
        command = [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-C",
            str(target),
            "commit",
            "-m",
            message,
        ]
        code, output, _, timed_out = _run_git_bounded(command, environment)
        if timed_out:
            return "Error: Git commit timed out."
        if code != 0:
            return f"Git commit failed: {output.strip()[:1000]}"
        return output.strip() or "Committed changes."
    except (OSError, ValueError) as exc:
        return f"Error creating Git commit: {exc}"


@tool(args_schema=CreatePullRequestInput)
def create_pull_request(
    title: str, body: str = "", base: str | None = None, directory: str = "."
) -> str:
    """Create a pull request with the GitHub CLI when it is available."""
    try:
        _, target, _, environment = _git_repository_root(directory)
        command = ["gh", "pr", "create", "--title", title, "--body", body]
        if base:
            command.extend(["--base", base])
        completed = subprocess.run(
            command,
            cwd=str(target),
            env=environment,
            capture_output=True,
            text=True,
            timeout=_GIT_INSPECTION_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip()
            return f"Pull request creation failed: {error[:1000]}"
        return completed.stdout.strip() or "Pull request created."
    except FileNotFoundError:
        return "Error: GitHub CLI (gh) is not installed."
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return f"Error creating pull request: {exc}"


@tool(args_schema=PreviewWorkspacePatchInput)
def preview_workspace_patch(file_path: str, old_text: str, new_text: str) -> str:
    """Preview the exact diff for a unique text replacement without changing the file."""
    try:
        target = SandboxManager.validate_path(file_path)
        _, _, diff = _exact_patch(target, file_path, old_text, new_text)
        return diff or "Patch preview: no changes."
    except (ValueError, OSError) as exc:
        return f"Error previewing patch: {exc}"


@tool(args_schema=ApplyWorkspacePatchInput)
def apply_workspace_patch(file_path: str, old_text: str, new_text: str) -> str:
    """Apply a unique, preconditioned workspace text patch after creating a snapshot backup."""
    try:
        if _path_contains_symlink(file_path):
            return "Error applying patch: Symlink paths are not modified."
        SandboxManager.validate_path(file_path)
        return _apply_workspace_patch_safely(file_path, old_text, new_text)
    except (ValueError, OSError) as exc:
        return f"Error applying patch: {exc}"


@tool(args_schema=RenameWorkspaceFileInput)
def rename_workspace_file(source_path: str, destination_path: str) -> str:
    """Rename a workspace file after validating both paths and preserving a snapshot."""
    try:
        if _path_contains_symlink(source_path) or _path_contains_symlink(
            destination_path
        ):
            return "Error renaming workspace file: Symlink paths are not allowed."
        SandboxManager.validate_path(source_path)
        SandboxManager.validate_path(destination_path)
        return _rename_workspace_file_safely(source_path, destination_path)
    except (ValueError, OSError) as exc:
        return f"Error renaming workspace file: {exc}"


@tool(args_schema=DeleteWorkspaceFileInput)
def delete_workspace_file(file_path: str) -> str:
    """Delete a workspace file only after approval, creating a recoverable snapshot first."""
    try:
        if _path_contains_symlink(file_path):
            return "Error deleting workspace file: Symlink paths are not allowed."
        SandboxManager.validate_path(file_path)
        _delete_workspace_file_safely(file_path)
        return f"Success: Deleted '{file_path}' after creating a recoverable snapshot."
    except (ValueError, OSError) as exc:
        return f"Error deleting workspace file: {exc}"
