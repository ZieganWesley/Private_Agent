"""Unit tests for the typed settings object and its coercion helpers."""

import ast
import copy
import pathlib
import time
import warnings
from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType
from typing import get_type_hints

import pytest

import config
from config.settings import (
    DEFAULT_PATHS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_PATH,
    Settings,
    configured_bool,
    configured_float,
    configured_int,
    configured_int_range,
    configured_nonnegative_int,
    configured_path,
    positive_float,
    positive_int,
    resolve_config_values,
)
from hardware import normalize_acceleration_mode

NON_SETTINGS_EXPORTS = {
    "APP_CONFIG",
    "CONFIG_FILE_PATH",
    "CONFIG_PATH_ENV_VAR",
    "DEFAULT_CONFIG",
    "DEFAULT_PATHS",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_SYSTEM_PROMPT_PATH",
    "SETTINGS",
}


def build(config_mapping=None):
    """Build settings from a raw (unmerged) config mapping."""
    return Settings.from_config(config_mapping or {}, default_paths=DEFAULT_PATHS)


# --------------------------------------------------------------- schema drift


def _settings_assignments():
    """Return {constant_name: (field_name, value)} for ``NAME = SETTINGS.field``."""
    tree = ast.parse(pathlib.Path(config.__file__).read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "SETTINGS"
        ):
            found[target.id] = (value.attr, node)
    return found


def test_every_settings_field_is_exposed_by_config():
    assignments = _settings_assignments()
    field_names = {field.name for field in fields(Settings)}

    assert {field for field, _node in assignments.values()} == field_names
    for constant, (field_name, _node) in assignments.items():
        assert constant == field_name.upper()
        assert getattr(config, constant) == getattr(config.SETTINGS, field_name)


def test_config_exposes_no_settings_constants_outside_the_dataclass():
    derived = {
        name
        for name, value in vars(config).items()
        if name.isupper()
        and name not in NON_SETTINGS_EXPORTS
        and not name.startswith("_")
    }

    assert derived == {constant for constant in _settings_assignments()}


def test_settings_matches_the_module_constants():
    assert build(config.APP_CONFIG) == config.SETTINGS


# ------------------------------------------------------------------ dataclass


def test_settings_is_frozen_and_typed():
    settings = build()

    with pytest.raises(FrozenInstanceError):
        settings.max_tool_iterations = 1

    hints = get_type_hints(Settings)
    assert hints["max_tool_iterations"] is int
    assert hints["model_temperature"] is float
    assert hints["online_permission_override"] is bool
    assert hints["max_output_tokens"] is int


def test_default_settings_come_from_the_shipped_defaults():
    settings = build()
    expected = Settings.from_config(config.DEFAULT_CONFIG, default_paths=DEFAULT_PATHS)

    assert settings == expected

    defaults = config.DEFAULT_CONFIG
    assert settings.max_output_tokens == defaults["agent"]["max_output_tokens"]
    assert settings.max_tool_iterations == defaults["agent"]["max_tool_iterations"]
    assert settings.agent_permission_mode == "auto"
    assert settings.web_research_consent == "ask"
    assert settings.thinking_effort_default == "medium"
    assert settings.chunking_strategy == "character"
    assert settings.chunking_unit == "chars"
    assert settings.jpeg_quality == defaults["media"]["jpeg_quality"]
    assert settings.enable_web_research is True
    assert settings.override_tool_list == frozenset()
    assert settings.code_task_test_commands == []
    assert settings.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert settings.default_db_path == str(
        pathlib.Path(defaults["paths"]["database"]).expanduser()
    )


def test_packaged_system_prompt_is_readable():
    assert DEFAULT_SYSTEM_PROMPT_PATH.exists()
    assert DEFAULT_SYSTEM_PROMPT
    assert (
        DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        == DEFAULT_SYSTEM_PROMPT
    )


# --------------------------------------------------------- coercion behaviour


@pytest.mark.parametrize(
    ("value", "default", "expected_suffix"),
    [
        ("~/logs", "/tmp", "logs"),
        ("relative/dir", "/tmp", "relative/dir"),
        (None, "~/fallback", "fallback"),
        ("", "~/fallback", "fallback"),
        (pathlib.Path("/abs/path"), "/tmp", None),
    ],
)
def test_configured_path_expands_and_anchors(value, default, expected_suffix):
    resolved = configured_path(value, default)
    expected = (
        "/abs/path"
        if expected_suffix is None
        else str(pathlib.Path.home() / expected_suffix)
    )

    assert resolved == expected
    assert pathlib.Path(resolved).is_absolute()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(7, 7), ("7", 7), (0, 15), (-3, 15), ("abc", 15), (None, 15), (True, 1)],
)
def test_positive_int(value, expected):
    assert positive_int({"max_tool_iterations": value}, "max_tool_iterations", 15) == (
        expected
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, 0.5), ("0.75", 0.75), (0, 0.2), (-1, 0.2), ("bad", 0.2), (None, 0.2)],
)
def test_positive_float(value, expected):
    assert positive_float({"model_temperature": value}, "model_temperature", 0.2) == (
        expected
    )


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ({"chunk_size": 400}, 400),
        ({"chunk_size": "400"}, 400),
        ({"chunk_size": 0}, 1200),
        ({"chunk_size": -1}, 1200),
        ({"chunk_size": "x"}, 1200),
        ({}, 1200),
        ("bad", 1200),
        ([], 1200),
        (None, 1200),
    ],
)
def test_configured_int(section, expected):
    assert configured_int({"chunking": section}, "chunking", "chunk_size", 1200) == (
        expected
    )


def test_configured_nonnegative_int_accepts_zero():
    assert configured_nonnegative_int({"a": {"n": 0}}, "a", "n", 5) == 0
    assert configured_nonnegative_int({"a": {"n": -1}}, "a", "n", 5) == 5
    assert configured_nonnegative_int({"a": "bad"}, "a", "n", 5) == 5


def test_configured_int_range_clamps_out_of_range_values():
    config_mapping = {"chunking": {"semantic_breakpoint_percentile": 250}}
    assert (
        configured_int_range(
            config_mapping, "chunking", "semantic_breakpoint_percentile", 95, 1, 100
        )
        == 95
    )
    config_mapping = {"chunking": {"semantic_breakpoint_percentile": 42}}
    assert (
        configured_int_range(
            config_mapping, "chunking", "semantic_breakpoint_percentile", 95, 1, 100
        )
        == 42
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        (" YES ", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("off", False),
        ("0", False),
        ("bogus", True),
        (1, True),
        (None, True),
    ],
)
def test_configured_bool(value, expected):
    assert configured_bool({"a": {"flag": value}}, "a", "flag", True) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.5, 1.5), ("2.5", 2.5), (0, 3.0), (-1, 3.0), ("bad", 3.0), (None, 3.0)],
)
def test_configured_float(value, expected):
    assert configured_float({"a": {"f": value}}, "a", "f", 3.0) == expected


# ------------------------------------------------------------- enum-ish fields


def test_permission_mode_and_effort_are_normalised():
    assert build({"agent_permission_mode": "FULL"}).agent_permission_mode == "full"
    assert build({"agent_permission_mode": "yolo"}).agent_permission_mode == "auto"
    assert build({"thinking_effort_default": "HIGH"}).thinking_effort_default == "high"
    assert (
        build({"thinking_effort_default": "insane"}).thinking_effort_default == "medium"
    )
    assert build({"web_research_consent": "Session"}).web_research_consent == "session"
    assert build({"web_research_consent": "always"}).web_research_consent == "ask"
    assert build({"logging": {"level": "debug"}}).debug_log_level == "DEBUG"
    assert build({"logging": {"level": "silly"}}).debug_log_level == "INFO"


def test_search_fallback_defaults_on_and_can_be_switched_off():
    defaults = build({})
    assert defaults.search_provider == "auto"
    assert defaults.searxng_url == ""
    assert defaults.search_fallback is True
    assert build({"network": {"search_fallback": "false"}}).search_fallback is False
    assert build({"network": {"search_fallback": False}}).search_fallback is False
    assert build({"network": {"search_fallback": "bogus"}}).search_fallback is True


def test_boolean_flags_accept_textual_spellings():
    assert build({"enable_web_research": "no"}).enable_web_research is False
    assert build({"enable_web_research": "yes"}).enable_web_research is True
    assert build({"enable_web_research": 0}).enable_web_research is False
    assert build({"thinking_toggle_default": "on"}).thinking_toggle_default is True
    assert build({"thinking_toggle_default": "off"}).thinking_toggle_default is False
    assert build({"DEBUG_LOG_ENABLED": "1"}).debug_log_enabled is True
    assert build({"DEBUG_LOG_ENABLED": "0"}).debug_log_enabled is False
    assert build({"DEBUG_LOG_ENABLED": 1}).debug_log_enabled is True


def test_hardware_acceleration_mode_is_validated():
    assert build({"hardware_acceleration": "GPU"}).hardware_acceleration_mode == (
        normalize_acceleration_mode("gpu")
    )
    assert build({"hardware_acceleration": "nonsense"}).hardware_acceleration_mode == (
        normalize_acceleration_mode("auto")
    )


def test_chunking_separators_default_and_custom():
    assert build().chunking_separators == ("\n\n", "\n", ". ", " ")
    assert build({"chunking": {"separators": ["|", "||"]}}).chunking_separators == (
        "|",
        "||",
    )
    for bad in ("nope", [], [1, 2], {"a": 1}):
        assert build({"chunking": {"separators": bad}}).chunking_separators == (
            "\n\n",
            "\n",
            ". ",
            " ",
        )


def test_override_tool_list_keeps_only_named_tools():
    settings = build(
        {"tools": {"override_tool_list": ["read_file", 3, "  ", "write_file"]}}
    )
    assert settings.override_tool_list == frozenset({"read_file", "write_file"})

    settings = build({"tools": {"override_tool_list": "read_file"}})
    assert settings.override_tool_list == frozenset()


def test_mcp_settings_tolerate_bad_shapes():
    servers = {"one": {"command": "run"}}
    settings = build({"mcpServers": servers, "mcp_auto_approve_tools": ["a", 2]})
    assert settings.mcp_servers == servers
    assert settings.mcp_auto_approve_tools == {"a", "2"}

    settings = build({"mcpServers": ["bad"], "mcp_auto_approve_tools": "a"})
    assert settings.mcp_servers == {}
    assert settings.mcp_auto_approve_tools == set()


def test_system_prompt_falls_back_when_blank():
    assert build({"agent": {"system_prompt": "   "}}).system_prompt == (
        DEFAULT_SYSTEM_PROMPT
    )
    assert build({"agent": {"system_prompt": "custom"}}).system_prompt == "custom"


# ------------------------------------------------------ code task test commands


def test_code_task_test_commands_keep_valid_entries():
    settings = build(
        {"code_tasks": {"test_commands": [["pytest", "-q"], ["ruff", "check", "."]]}}
    )

    assert settings.code_task_test_commands == [
        ["pytest", "-q"],
        ["ruff", "check", "."],
    ]


def test_code_task_test_commands_drop_invalid_entries_with_warning():
    raw = [
        ["pytest", "-q"],
        "not-a-list",
        [],
        ["ok", 5],
        ["x" * 2049],
    ]
    with pytest.warns(
        RuntimeWarning, match="Ignoring invalid code_tasks.test_commands"
    ):
        settings = build({"code_tasks": {"test_commands": raw}})

    assert settings.code_task_test_commands == [["pytest", "-q"]]


def test_code_task_test_commands_reject_non_list_with_single_warning():
    with pytest.warns(RuntimeWarning, match="expected an array of argument arrays"):
        settings = build({"code_tasks": {"test_commands": "pytest"}})

    assert settings.code_task_test_commands == []


def test_code_task_test_commands_warn_when_truncating():
    raw = [["cmd", str(index)] for index in range(20)]
    with pytest.warns(RuntimeWarning, match="first 16 code_tasks.test_commands"):
        settings = build({"code_tasks": {"test_commands": raw}})

    assert len(settings.code_task_test_commands) == 16
    assert settings.code_task_test_commands[-1] == ["cmd", "15"]


# -------------------------------------------------------------- robustness


@pytest.mark.parametrize(
    "broken",
    [
        {"agent": "not-a-dict", "memory": [], "paths": "nope"},
        {"chunking": [], "tools": {"override_tool_list": None}, "code_tasks": []},
        {"mcpServers": "x", "logging": 5, "media": {"jpeg_quality": "huge"}},
    ],
)
def test_malformed_sections_fall_back_to_defaults(broken):
    default = build()

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        settings = build(broken)

    assert settings.agent_permission_mode == default.agent_permission_mode
    assert settings.max_tool_iterations == default.max_tool_iterations
    assert settings.jpeg_quality == default.jpeg_quality
    assert settings.chunking_separators == default.chunking_separators
    assert settings.mcp_servers == default.mcp_servers


def test_resolution_does_not_mutate_its_input():
    raw = {
        "agent": {"max_tool_iterations": 3.7},
        "tools": {"override_tool_list": ["read_file", 1]},
        "chunking": {"separators": ["|"]},
        "code_tasks": {"test_commands": [["pytest"]]},
    }
    snapshot = copy.deepcopy(raw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        resolve_config_values(raw, default_paths=DEFAULT_PATHS)

    assert raw == snapshot


def test_resolution_accepts_read_only_mappings():
    raw = MappingProxyType(
        {"max_tool_iterations": 4, "agent": MappingProxyType({"max_output_tokens": 77})}
    )

    settings = Settings.from_config(raw, default_paths=DEFAULT_PATHS)

    assert settings.max_tool_iterations == 4
    assert settings.max_output_tokens == 77


def test_resolution_is_deterministic_and_fast():
    started = time.perf_counter()
    for _ in range(200):
        Settings.from_config(config.DEFAULT_CONFIG, default_paths=DEFAULT_PATHS)
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0


def test_values_are_independent_between_instances():
    first = build({"tools": {"override_tool_list": ["a"]}})
    second = build({"tools": {"override_tool_list": ["b"]}})

    assert first.override_tool_list == frozenset({"a"})
    assert second.override_tool_list == frozenset({"b"})


def test_config_compatibility_wrappers_read_the_current_module_config(monkeypatch):
    monkeypatch.setattr(
        config,
        "APP_CONFIG",
        {"max_tool_iterations": "9", "agent": {"max_output_tokens": "1234"}},
    )

    assert config._configured_int("agent", "max_output_tokens", 1) == 1234
    assert config._positive_int("max_tool_iterations", 15) == 9
    assert config._configured_bool("agent", "missing", True) is True
    assert config._configured_int_range("agent", "max_output_tokens", 1, 1, 100) == 1
    assert config._configured_nonnegative_int("agent", "max_output_tokens", 0) == 1234
    assert config._configured_float("agent", "ratio", 0.5) == 0.5
    assert config._configured_path("~/x", "/tmp").endswith(str(pathlib.Path("x")))


def test_embedding_provider_accepts_an_openai_compatible_endpoint():
    settings = Settings.from_config(
        {
            "rag": {
                "embedding_provider": "openai-compatible",
                "embedding_base_url": "http://127.0.0.1:8000/v1/",
            }
        },
        default_paths=DEFAULT_PATHS,
    )

    assert settings.embedding_provider == "openai-compatible"
    assert settings.embedding_base_url == "http://127.0.0.1:8000/v1/"


def test_unknown_embedding_provider_falls_back_to_ollama():
    settings = Settings.from_config(
        {"rag": {"embedding_provider": "some-cloud"}},
        default_paths=DEFAULT_PATHS,
    )

    assert settings.embedding_provider == "ollama"
    assert settings.embedding_base_url == "http://127.0.0.1:1234/v1"


def test_rerank_settings_expose_the_planned_config_surface():
    settings = Settings.from_config({}, default_paths=DEFAULT_PATHS)

    assert settings.rerank_mode == "auto"
    assert settings.rerank_model == ""
    assert settings.rerank_timeout_seconds == 2.5
    assert settings.rerank_top_n == 20
    assert settings.rerank_max_candidates == 50
    assert settings.rerank_local_threshold == 20
    assert settings.rerank_cache_entries == 128
    assert settings.rerank_breaker_failures == 3
    assert settings.rerank_breaker_cooldown_seconds == 60.0


def test_rerank_settings_read_configured_values_and_guard_invalid_ones():
    settings = Settings.from_config(
        {
            "rag": {
                "rerank_mode": "LOCAL",
                "rerank_model": "bge-reranker",
                "rerank_timeout_seconds": 0,
                "rerank_top_n": "8",
                "rerank_max_candidates": 12,
                "rerank_local_threshold": 0,
                "rerank_cache_entries": 32,
                "rerank_breaker_failures": 4,
                "rerank_breaker_cooldown_seconds": 15.5,
            }
        },
        default_paths=DEFAULT_PATHS,
    )

    assert settings.rerank_mode == "local"
    assert settings.rerank_model == "bge-reranker"
    assert settings.rerank_timeout_seconds == 2.5
    assert settings.rerank_top_n == 8
    assert settings.rerank_max_candidates == 12
    assert settings.rerank_local_threshold == 0
    assert settings.rerank_cache_entries == 32
    assert settings.rerank_breaker_failures == 4
    assert settings.rerank_breaker_cooldown_seconds == 15.5
