from private_agent.doctor import analyze_machine, check_dependencies, recommend_config


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


class _FakeConsole:
    def __init__(self, answer):
        self.answer = answer
        self.output = []

    def print(self, *args, **kwargs):
        self.output.append(" ".join(str(a) for a in args))

    def input(self, prompt=""):
        self.output.append(prompt)
        return self.answer


def test_doctor_interactive_recommends_on_confirmation(monkeypatch):
    from private_agent import doctor

    fake = _FakeConsole("y")
    monkeypatch.setattr(doctor, "console", fake)
    monkeypatch.setattr(doctor, "analyze_machine", lambda: {"cpu_count": 4, "memory_gb": 8.0, "platform": "x", "python": "3", "disk_free_gb": 10.0, "home": "/tmp"})
    result = doctor.doctor_interactive()
    assert result == 0
    joined = "\n".join(fake.output)
    assert "Recommended profile" in joined


def test_doctor_interactive_manual_fallback_on_decline(monkeypatch):
    from private_agent import doctor

    fake = _FakeConsole("n")
    monkeypatch.setattr(doctor, "console", fake)
    monkeypatch.setattr(doctor, "analyze_machine", lambda: {"cpu_count": 2, "memory_gb": 4.0, "platform": "x", "python": "3", "disk_free_gb": 10.0, "home": "/tmp"})
    result = doctor.doctor_interactive()
    assert result == 0
    joined = "\n".join(fake.output)
    assert "Manual config changes" in joined
    assert ".private_agent.conf" in joined
