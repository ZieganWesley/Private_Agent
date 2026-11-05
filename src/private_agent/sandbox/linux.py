"""Workspace path controls and Linux command risk helpers."""

import os
import pathlib
import shlex
import shutil
import uuid

SNAPSHOT_DIR = pathlib.Path(".agent_snapshots").resolve()


class SandboxManager:
    root_dir: pathlib.Path = pathlib.Path.cwd().resolve()

    @classmethod
    def set_root(cls, path_str: str):
        real_root = os.path.realpath(path_str)
        p = pathlib.Path(real_root)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        cls.root_dir = p.resolve()

    @classmethod
    def validate_path(cls, file_path: str) -> pathlib.Path:
        try:
            p = pathlib.Path(file_path)
            raw_target = p if p.is_absolute() else (cls.root_dir / file_path)

            # Strengthen path validation by explicitly resolving all symlinks via os.path.realpath
            real_target_str = os.path.realpath(str(raw_target))
            real_root_str = os.path.realpath(str(cls.root_dir))

            target = pathlib.Path(real_target_str)
            root = pathlib.Path(real_root_str)

            if not target.is_relative_to(root):
                raise PermissionError
            return target
        except Exception:
            raise PermissionError(
                f"Security Violation: Access denied outside workspace root ({cls.root_dir})"
            ) from None


def create_hitl_snapshot(target_path: pathlib.Path):
    if target_path.exists():
        root = SandboxManager.root_dir.resolve()
        source = target_path.resolve()
        if not source.is_relative_to(root):
            raise PermissionError(
                "Cannot create a snapshot outside the active workspace."
            )
        snapshot_dir = root / ".agent_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        if not snapshot_dir.resolve().is_relative_to(root):
            raise PermissionError(
                "Workspace snapshot directory cannot escape the workspace."
            )
        backup_name = f"{target_path.name}_{uuid.uuid4().hex}.bak"
        shutil.copy2(source, snapshot_dir / backup_name)


_ELEVATED_PROGRAMS = frozenset({"sudo", "su", "doas", "pkexec", "runuser"})
_DESTRUCTIVE_PROGRAMS = frozenset({"mkfs", "dd", "shred", "wipefs", "fdisk", "parted"})
_OWNERSHIP_PROGRAMS = frozenset({"chown", "chgrp"})
_WRITING_PROGRAMS = frozenset(
    {"rm", "rmdir", "mv", "tee", "touch", "truncate", "chmod"}
)
# Only the destination (last argument) of these programs is written to.
_COPYING_PROGRAMS = frozenset({"cp", "ln", "install", "rsync"})
_WRAPPER_PROGRAMS = frozenset(
    {"env", "nohup", "nice", "time", "command", "exec", "xargs"}
)
_SHELL_PROGRAMS = frozenset({"sh", "bash", "dash", "zsh"})
_COMMAND_SEPARATORS = frozenset({";", "&", "&&", "|", "||", "|&", "(", ")", "()"})
_REDIRECT_OPERATORS = frozenset({">", ">>", ">|", "&>", "&>>", ">&"})
_HARMLESS_DEVICES = frozenset(
    {"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty", "/dev/zero"}
)
_SYSTEM_PATHS = (
    "/etc",
    "/var",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/boot",
    "/root",
    "/sys",
    "/proc",
    "/dev",
)
_FORK_BOMB = ":(){:|:&};:"


def _tokenize_shell_command(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        # Unbalanced quotes: fall back to whitespace splitting so the check
        # still sees the program names instead of silently passing.
        return command.split()


def _split_commands(tokens: list[str]) -> tuple[list[list[str]], list[str]]:
    """Split tokens into simple commands and collect redirect targets."""
    commands: list[list[str]] = [[]]
    redirect_targets: list[str] = []
    expect_target = False
    for token in tokens:
        if expect_target:
            redirect_targets.append(token)
            expect_target = False
        elif token in _COMMAND_SEPARATORS:
            commands.append([])
        elif token in _REDIRECT_OPERATORS:
            expect_target = True
        else:
            commands[-1].append(token)
    return [cmd for cmd in commands if cmd], redirect_targets


def _program_and_args(words: list[str]) -> tuple[str, list[str]]:
    """Skip `VAR=value` prefixes and wrappers such as `env` to find the program."""
    index = 0
    while index < len(words):
        word = words[index]
        name = os.path.basename(word)
        if "=" in word and not word.startswith("-") and name == word:
            index += 1
        elif name in _WRAPPER_PROGRAMS:
            index += 1
            while index < len(words) and words[index].startswith("-"):
                index += 1
        else:
            return name, words[index + 1 :]
    return "", []


def _short_flags(args: list[str]) -> set[str]:
    flags: set[str] = set()
    for arg in args:
        if arg == "--":
            break
        if arg.startswith("-") and not arg.startswith("--"):
            flags.update(arg[1:])
    return flags


def _is_system_path(arg: str) -> bool:
    if not arg.startswith("/"):
        return False
    # normpath keeps a leading "//", so collapse it before comparing.
    path = "/" + os.path.normpath(arg).lstrip("/")
    if path in _HARMLESS_DEVICES:
        return False
    return path == "/" or any(
        path == root or path.startswith(root + "/") for root in _SYSTEM_PATHS
    )


def _is_risky_simple_command(program: str, args: list[str]) -> bool:
    if program in _ELEVATED_PROGRAMS:
        return True
    if program in _DESTRUCTIVE_PROGRAMS or program.startswith("mkfs."):
        return True
    if program in _OWNERSHIP_PROGRAMS:
        return True
    if program == "rm" and (_short_flags(args) & {"r", "R"} or "--recursive" in args):
        return True
    if program == "chmod" and (
        "R" in _short_flags(args)
        or "--recursive" in args
        or any(arg.lstrip("0") == "777" or arg in {"a+rwx", "ugo+rwx"} for arg in args)
    ):
        return True
    if program == "find" and (
        "-delete" in args
        or any(
            flag in {"-exec", "-execdir", "-ok", "-okdir"}
            and index + 1 < len(args)
            and os.path.basename(args[index + 1]) in {"rm", "shred"}
            for index, flag in enumerate(args)
        )
    ):
        return True
    if program in _COPYING_PROGRAMS:
        operands = [arg for arg in args if not arg.startswith("-")]
        return bool(operands) and _is_system_path(operands[-1])
    return program in _WRITING_PROGRAMS and any(_is_system_path(arg) for arg in args)


def evaluate_shell_command(command: str) -> bool:
    """Return True when a command looks risky enough to warn about.

    This only adds a warning to the approval prompt; the OS sandbox is the real
    protection. The command is tokenized with `shlex`, so quoted text such as
    `echo "rm -rf /"` and harmless reads such as `ls /usr` are not flagged.
    """
    if _FORK_BOMB in command.replace(" ", ""):
        return True
    commands, redirect_targets = _split_commands(_tokenize_shell_command(command))
    for target in redirect_targets:
        if _is_system_path(target):
            return True
    for words in commands:
        program, args = _program_and_args(words)
        if program in _SHELL_PROGRAMS and "-c" in args:
            script_index = args.index("-c") + 1
            if script_index < len(args) and evaluate_shell_command(args[script_index]):
                return True
        elif program and _is_risky_simple_command(program, args):
            return True
    return False
