"""Generated defaults live in one module and feed the model (K9/14.3)."""

from __future__ import annotations

import ast
import json
import pathlib

import config.settings as settings_module
from config import DEFAULT_CONFIG
from config.defaults import DEFAULT_PATHS
from config.settings import Settings, default_for

CONFIGURED_HELPERS = {
    "configured_str",
    "configured_int",
    "configured_bool",
    "configured_float",
    "configured_nonnegative_int",
    "configured_int_range",
    "configured_choice",
}


def test_defaults_are_defined_once() -> None:
    assert settings_module.DEFAULT_CONFIG is DEFAULT_CONFIG
    assert settings_module.DEFAULT_PATHS is DEFAULT_PATHS
    assert DEFAULT_CONFIG["paths"] == DEFAULT_PATHS
    assert DEFAULT_CONFIG["paths"] is not DEFAULT_PATHS


def test_default_for_reads_the_shipped_defaults() -> None:
    assert default_for("agent", "max_tool_iterations") == 15
    assert default_for("tools", "max_read_file_bytes") == 1048576
    assert default_for("media", "jpeg_quality") == DEFAULT_CONFIG["media"]["jpeg_quality"]


def test_settings_source_has_no_literal_defaults_for_configured_keys() -> None:
    tree = ast.parse(pathlib.Path(settings_module.__file__).read_text())
    literal_defaults = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in CONFIGURED_HELPERS or len(node.args) < 4:
            continue
        section, default = node.args[1], node.args[3]
        if (
            isinstance(section, ast.Constant)
            and section.value in DEFAULT_CONFIG
            and isinstance(default, ast.Constant)
        ):
            literal_defaults.append(node.lineno)
    assert literal_defaults == []


def test_the_generated_file_parses_back_to_the_same_model() -> None:
    payload = json.loads(json.dumps(DEFAULT_CONFIG, indent=4))
    from_file = Settings.from_config(payload, default_paths=payload["paths"])
    from_model = Settings.from_config(
        DEFAULT_CONFIG, default_paths=DEFAULT_CONFIG["paths"]
    )

    assert from_file == from_model
    assert from_file.max_tool_iterations == DEFAULT_CONFIG["agent"]["max_tool_iterations"]
