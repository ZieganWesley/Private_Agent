"""Config can only tighten sandbox and RAG lists, never widen them (K10/14.11)."""

from __future__ import annotations

import config
from config import DEFAULT_CONFIG
from config.settings import Settings


def _settings(sandbox: dict | None = None, rag: dict | None = None) -> Settings:
    return Settings.from_config(
        {"sandbox": sandbox or {}, "rag": rag or {}},
        default_paths=DEFAULT_CONFIG["paths"],
    )


def test_hardening_lists_start_empty() -> None:
    settings = _settings()

    assert settings.sandbox_extra_risky_programs == []
    assert settings.sandbox_extra_system_paths == []
    assert settings.rag_excluded_file_types == []


def test_extra_risky_programs_are_validated() -> None:
    settings = _settings(
        {"extra_risky_programs": ["Wipe", "wipe", "rm -rf", "a/b", "", 5, "x" * 65]}
    )

    assert settings.sandbox_extra_risky_programs == ["wipe", "wipe"]


def test_extra_system_paths_must_be_absolute_and_normalized() -> None:
    settings = _settings(
        {
            "extra_system_paths": [
                "/opt/protected",
                "relative/path",
                "/opt/shared/../protected",
                "",
                5,
            ]
        }
    )

    assert settings.sandbox_extra_system_paths == ["/opt/protected", "/opt/protected"]


def test_excluded_rag_types_only_remove_supported_types() -> None:
    settings = _settings(rag={"excluded_file_types": ["MD", ".txt", 5, ""]})

    assert settings.rag_excluded_file_types == [".md", ".txt"]

    from rag import indexing

    assert ".md" in indexing.SUPPORTED_EXTENSIONS
    assert ".xyz" not in indexing.SUPPORTED_EXTENSIONS
    assert indexing.SQLITE_EXTENSIONS <= indexing.SUPPORTED_EXTENSIONS


def test_sandbox_lists_are_built_on_top_of_the_builtins(monkeypatch) -> None:
    from sandbox import linux

    assert frozenset(config.SANDBOX_EXTRA_RISKY_PROGRAMS) == linux._EXTRA_RISKY_PROGRAMS
    assert {"/etc", "/usr", "/proc"} <= set(linux._SYSTEM_PATHS)
    assert {"sudo", "su", "mkfs", "dd"} <= linux._ELEVATED_PROGRAMS | (
        linux._DESTRUCTIVE_PROGRAMS
    )

    monkeypatch.setattr(linux, "_EXTRA_RISKY_PROGRAMS", frozenset({"mywipe"}))
    assert linux._is_risky_simple_command("mywipe", [])
    assert not linux._is_risky_simple_command("touch", ["notes.txt"])

    monkeypatch.setattr(
        linux, "_SYSTEM_PATHS", linux._SYSTEM_PATHS + ("/opt/protected",)
    )
    assert linux._is_system_path("/opt/protected/data")
    assert not linux._is_system_path("/opt/other/data")
    assert linux._is_risky_simple_command("touch", ["/opt/protected/data"])


def test_module_constants_mirror_settings() -> None:
    assert (
        config.SETTINGS.sandbox_extra_risky_programs
        == config.SANDBOX_EXTRA_RISKY_PROGRAMS
    )
    assert (
        config.SETTINGS.sandbox_extra_system_paths == config.SANDBOX_EXTRA_SYSTEM_PATHS
    )
    assert config.SETTINGS.rag_excluded_file_types == config.RAG_EXCLUDED_FILE_TYPES
