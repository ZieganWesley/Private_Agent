"""The sandbox resource limits come from settings with safe floors (K2/14.4)."""

from __future__ import annotations

import config
from config import DEFAULT_CONFIG
from config.settings import Settings

GIB = 1024**3
MIB = 1024**2


def _settings(code_tasks: dict | None = None) -> Settings:
    return Settings.from_config(
        {"code_tasks": code_tasks or {}}, default_paths=DEFAULT_CONFIG["paths"]
    )


def test_sandbox_limits_default_to_four_gib_and_existing_limits() -> None:
    settings = _settings()

    assert settings.code_task_sandbox_cpu_seconds == 180
    assert settings.code_task_sandbox_address_space_bytes == 4 * GIB
    assert settings.code_task_sandbox_process_count == 128
    assert settings.code_task_sandbox_file_size_bytes == 512 * MIB


def test_default_config_and_generated_file_agree_on_sandbox_limits() -> None:
    raw = DEFAULT_CONFIG["code_tasks"]

    assert raw["sandbox_cpu_seconds"] == 180
    assert raw["sandbox_address_space_bytes"] == 4 * GIB
    assert raw["sandbox_process_count"] == 128
    assert raw["sandbox_file_size_bytes"] == 512 * MIB


def test_sandbox_limits_can_be_raised_from_config() -> None:
    settings = _settings(
        {
            "sandbox_cpu_seconds": 600,
            "sandbox_address_space_bytes": 8 * GIB,
            "sandbox_process_count": 512,
            "sandbox_file_size_bytes": 2 * GIB,
        }
    )

    assert settings.code_task_sandbox_cpu_seconds == 600
    assert settings.code_task_sandbox_address_space_bytes == 8 * GIB
    assert settings.code_task_sandbox_process_count == 512
    assert settings.code_task_sandbox_file_size_bytes == 2 * GIB


def test_sandbox_limits_below_floors_fall_back_to_defaults() -> None:
    settings = _settings(
        {
            "sandbox_cpu_seconds": 0,
            "sandbox_address_space_bytes": GIB - 1,
            "sandbox_process_count": 1,
            "sandbox_file_size_bytes": 0,
        }
    )

    assert settings.code_task_sandbox_cpu_seconds == 180
    assert settings.code_task_sandbox_address_space_bytes == 4 * GIB
    assert settings.code_task_sandbox_process_count == 128
    assert settings.code_task_sandbox_file_size_bytes == 512 * MIB


def test_sandbox_limits_reject_non_numeric_values() -> None:
    settings = _settings(
        {
            "sandbox_cpu_seconds": "lots",
            "sandbox_address_space_bytes": None,
            "sandbox_process_count": [],
            "sandbox_file_size_bytes": "1GiB",
        }
    )

    assert settings.code_task_sandbox_cpu_seconds == 180
    assert settings.code_task_sandbox_address_space_bytes == 4 * GIB
    assert settings.code_task_sandbox_process_count == 128
    assert settings.code_task_sandbox_file_size_bytes == 512 * MIB


def test_module_constants_mirror_settings() -> None:
    assert (
        config.SETTINGS.code_task_sandbox_cpu_seconds
        == config.CODE_TASK_SANDBOX_CPU_SECONDS
    )
    assert (
        config.SETTINGS.code_task_sandbox_address_space_bytes
        == config.CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES
    )
    assert (
        config.SETTINGS.code_task_sandbox_process_count
        == config.CODE_TASK_SANDBOX_PROCESS_COUNT
    )
    assert (
        config.SETTINGS.code_task_sandbox_file_size_bytes
        == config.CODE_TASK_SANDBOX_FILE_SIZE_BYTES
    )


def test_isolated_command_renders_configured_limits(monkeypatch, tmp_path) -> None:
    import code_tasks

    monkeypatch.setattr(code_tasks.platform, "system", lambda: "Linux")
    monkeypatch.setattr(code_tasks.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_CPU_SECONDS", 42)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES", 6 * GIB)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_PROCESS_COUNT", 256)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_FILE_SIZE_BYTES", 2 * MIB)

    isolated, _ = code_tasks._isolated_command(["python", "-V"], tmp_path)

    assert "--cpu=42" in isolated
    assert f"--as={6 * GIB}" in isolated
    assert "--nproc=256" in isolated
    assert f"--fsize={2 * MIB}" in isolated
