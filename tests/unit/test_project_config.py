import json

import pytest

from private_agent import config


def test_discover_project_config_paths(tmp_path):
    paths = config.discover_project_config_paths(tmp_path)
    assert paths == [
        tmp_path / ".private_agent" / "settings.json",
        tmp_path / ".private_agent" / "settings.local.json",
    ]


def test_load_project_config_missing_returns_empty(tmp_path):
    assert config.load_project_config(tmp_path) == {}


def test_load_project_config_merges_user_then_local(tmp_path):
    base = tmp_path / ".private_agent"
    base.mkdir()
    (base / "settings.json").write_text(
        json.dumps({"agent": {"model": "base"}, "tools": {"x": 1}}),
        encoding="utf-8",
    )
    (base / "settings.local.json").write_text(
        json.dumps({"agent": {"model": "local"}, "tools": {"y": 2}}),
        encoding="utf-8",
    )
    merged = config.load_project_config(tmp_path)
    assert merged["agent"]["model"] == "local"
    assert merged["tools"] == {"x": 1, "y": 2}


def test_load_project_config_malformed_warns(tmp_path):
    base = tmp_path / ".private_agent"
    base.mkdir()
    (base / "settings.json").write_text("{ bad json", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        assert config.load_project_config(tmp_path) == {}


def test_merge_project_config_overrides_base(tmp_path):
    base = tmp_path / ".private_agent"
    base.mkdir()
    (base / "settings.json").write_text(
        json.dumps({"agent": {"temperature": 0.9}}), encoding="utf-8"
    )
    result = config.merge_project_config({"agent": {"temperature": 0.1}}, tmp_path)
    assert result["agent"]["temperature"] == 0.9


def test_load_or_create_config_applies_project_config(tmp_path, monkeypatch):
    config_path = tmp_path / ".private_agent.conf"
    monkeypatch.setattr(config, "CONFIG_FILE_PATH", config_path)
    monkeypatch.chdir(tmp_path)
    project_dir = tmp_path / ".private_agent"
    project_dir.mkdir()
    (project_dir / "settings.json").write_text(
        json.dumps({"agent": {"streaming_output": False}}), encoding="utf-8"
    )
    settings = config.load_or_create_config()
    assert settings["agent"]["streaming_output"] is False
