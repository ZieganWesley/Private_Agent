import json

from diagnostics.doctor import (
    analyze_machine,
    check_config_file,
    check_dependencies,
    probe_endpoint,
    recommend_config,
)


def test_check_dependencies_structure():
    result = check_dependencies()
    assert "required" in result
    assert "optional" in result
    assert isinstance(result["required"], list)


def test_analyze_machine():
    info = analyze_machine()
    assert "platform" in info
    assert "cpu_count" in info
    assert info["cpu_count"] >= 1


def test_recommend_config():
    info = {"cpu_count": 8, "memory_gb": 16.0}
    rec = recommend_config(info)
    assert rec["profile"] in {"low", "medium", "high"}
    assert "settings" in rec


def test_check_config_file_accepts_a_valid_file(tmp_path):
    path = tmp_path / "private_config.json"
    path.write_text(
        json.dumps({"agent": {"max_tool_iterations": 5}, "models": {"endpoints": []}}),
        encoding="utf-8",
    )

    assert check_config_file(path) == []


def test_check_config_file_reports_malformed_json(tmp_path):
    path = tmp_path / "private_config.json"
    path.write_text('{"agent": {', encoding="utf-8")

    problems = check_config_file(path)

    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "not valid JSON" in problems[0].message
    assert "line 1" in problems[0].message


def test_check_config_file_reports_a_non_object_root(tmp_path):
    path = tmp_path / "private_config.json"
    path.write_text("[]", encoding="utf-8")

    problems = check_config_file(path)

    assert [problem.severity for problem in problems] == ["error"]
    assert "must contain a JSON object" in problems[0].message


def test_check_config_file_reports_wrong_section_and_value_types(tmp_path):
    path = tmp_path / "private_config.json"
    path.write_text(
        json.dumps(
            {
                "agent": ["not", "a", "table"],
                "models": {"endpoints": {}, "temperature": "hot"},
            }
        ),
        encoding="utf-8",
    )

    messages = [problem.message for problem in check_config_file(path)]

    assert "agent must be a JSON object, found list." in messages
    assert "models.endpoints must be list, found dict." in messages
    assert "models.temperature must be float, found str." in messages


def test_check_config_file_accepts_textual_booleans(tmp_path):
    path = tmp_path / "private_config.json"
    path.write_text(
        json.dumps({"models": {"raw_completions_enabled": "yes"}}), encoding="utf-8"
    )

    assert check_config_file(path) == []


def test_check_config_file_warns_when_the_file_is_missing(tmp_path):
    problems = check_config_file(tmp_path / "missing.json")

    assert [problem.severity for problem in problems] == ["warning"]
    assert "does not exist" in problems[0].message


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeClient:
    def __init__(self, status_code=None, error=None):
        self.status_code = status_code
        self.error = error
        self.closed = False

    def get(self, url):
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.status_code)

    def close(self):
        self.closed = True


def test_probe_endpoint_reports_reachability_and_closes_the_client():
    client = _FakeClient(status_code=200)

    assert probe_endpoint("http://127.0.0.1:1234/v1", client_factory=lambda **_: client)
    assert client.closed is True


def test_probe_endpoint_reports_http_status_and_failures():
    assert (
        probe_endpoint(
            "http://127.0.0.1:1234/v1", client_factory=lambda **_: _FakeClient(503)
        )
        == "HTTP 503"
    )
    assert (
        probe_endpoint(
            "http://127.0.0.1:1234/v1",
            client_factory=lambda **_: _FakeClient(error=OSError("refused")),
        )
        == "unreachable (OSError)"
    )


class _FakeConsole:
    def __init__(self, answer):
        self.answer = answer
        self.output = []

    def print(self, *args, **kwargs):
        self.output.append(" ".join(str(a) for a in args))

    def input(self, prompt=""):
        self.output.append(prompt)
        return self.answer


def _doctor_setup(monkeypatch, answer):
    from diagnostics import doctor

    fake = _FakeConsole(answer)
    monkeypatch.setattr(doctor, "console", fake)
    monkeypatch.setattr(doctor, "probe_endpoint", lambda base_url: "reachable")
    monkeypatch.setattr(doctor, "check_config_file", lambda path=None: [])
    monkeypatch.setattr(
        doctor,
        "analyze_machine",
        lambda: {
            "cpu_count": 8,
            "memory_gb": 16.0,
            "platform": "x",
            "python": "3",
            "disk_free_gb": 10.0,
            "home": "/tmp",
        },
    )
    return doctor, fake


def test_doctor_interactive_recommends_on_confirmation(monkeypatch):
    doctor, fake = _doctor_setup(monkeypatch, "y")

    assert doctor.doctor_interactive() == 0
    joined = "\n".join(fake.output)
    assert "Recommended profile" in joined


def test_doctor_interactive_manual_fallback_on_decline(monkeypatch):
    doctor, fake = _doctor_setup(monkeypatch, "n")

    assert doctor.doctor_interactive() == 0
    joined = "\n".join(fake.output)
    assert "Manual config changes" in joined
    assert "private_config.json" in joined


def test_run_doctor_fails_when_the_configuration_has_errors(monkeypatch):
    from diagnostics import doctor
    from diagnostics.doctor import ConfigProblem

    fake = _FakeConsole("n")
    monkeypatch.setattr(doctor, "console", fake)
    monkeypatch.setattr(doctor, "probe_endpoint", lambda base_url: "reachable")
    monkeypatch.setattr(
        doctor,
        "check_config_file",
        lambda path=None: [ConfigProblem("error", "broken")],
    )

    assert doctor.run_doctor() == 1
    assert "Fix the configuration errors" in "\n".join(fake.output)
