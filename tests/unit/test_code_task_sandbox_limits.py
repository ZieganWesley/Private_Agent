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


class _FakeResource:
    RLIM_INFINITY = -1
    RLIMIT_CPU = 1
    RLIMIT_AS = 2
    RLIMIT_NPROC = 3
    RLIMIT_FSIZE = 4

    def __init__(self, limits: dict[int, tuple[int, int]]) -> None:
        self._limits = limits

    def getrlimit(self, limit_id: int) -> tuple[int, int]:
        return self._limits[limit_id]


def _linux_sandbox(monkeypatch, resource_limits):
    import code_tasks

    monkeypatch.setattr(code_tasks.platform, "system", lambda: "Linux")
    monkeypatch.setattr(code_tasks.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(code_tasks, "resource", _FakeResource(resource_limits))
    return code_tasks


def test_isolated_command_clamps_limits_to_inherited_hard_limits(
    monkeypatch, tmp_path
) -> None:
    code_tasks = _linux_sandbox(
        monkeypatch,
        {
            _FakeResource.RLIMIT_CPU: (42, 42),
            _FakeResource.RLIMIT_AS: (2400000 * 1024, 2400000 * 1024),
            _FakeResource.RLIMIT_NPROC: (256, 256),
            _FakeResource.RLIMIT_FSIZE: (MIB, MIB),
        },
    )
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_CPU_SECONDS", 180)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES", 4 * GIB)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_PROCESS_COUNT", 512)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_FILE_SIZE_BYTES", 512 * MIB)

    isolated, _ = code_tasks._isolated_command(["python", "-V"], tmp_path)

    assert "--cpu=42" in isolated
    assert f"--as={2400000 * 1024}" in isolated
    assert "--nproc=256" in isolated
    assert f"--fsize={MIB}" in isolated


def test_isolated_command_keeps_limits_when_hard_limits_are_unlimited(
    monkeypatch, tmp_path
) -> None:
    unlimited = _FakeResource.RLIM_INFINITY
    code_tasks = _linux_sandbox(
        monkeypatch,
        {
            _FakeResource.RLIMIT_CPU: (unlimited, unlimited),
            _FakeResource.RLIMIT_AS: (unlimited, unlimited),
            _FakeResource.RLIMIT_NPROC: (unlimited, unlimited),
            _FakeResource.RLIMIT_FSIZE: (unlimited, unlimited),
        },
    )
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_CPU_SECONDS", 180)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES", 4 * GIB)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_PROCESS_COUNT", 128)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_FILE_SIZE_BYTES", 512 * MIB)

    isolated, _ = code_tasks._isolated_command(["python", "-V"], tmp_path)

    assert "--cpu=180" in isolated
    assert f"--as={4 * GIB}" in isolated
    assert "--nproc=128" in isolated
    assert f"--fsize={512 * MIB}" in isolated


def test_isolated_command_renders_configured_limits(monkeypatch, tmp_path) -> None:
    unlimited = _FakeResource.RLIM_INFINITY
    code_tasks = _linux_sandbox(
        monkeypatch,
        {
            _FakeResource.RLIMIT_CPU: (unlimited, unlimited),
            _FakeResource.RLIMIT_AS: (unlimited, unlimited),
            _FakeResource.RLIMIT_NPROC: (unlimited, unlimited),
            _FakeResource.RLIMIT_FSIZE: (unlimited, unlimited),
        },
    )
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_CPU_SECONDS", 42)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES", 6 * GIB)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_PROCESS_COUNT", 256)
    monkeypatch.setattr(code_tasks, "CODE_TASK_SANDBOX_FILE_SIZE_BYTES", 2 * MIB)

    isolated, _ = code_tasks._isolated_command(["python", "-V"], tmp_path)

    assert "--cpu=42" in isolated
    assert f"--as={6 * GIB}" in isolated
    assert "--nproc=256" in isolated
    assert f"--fsize={2 * MIB}" in isolated
