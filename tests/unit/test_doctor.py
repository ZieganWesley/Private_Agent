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
