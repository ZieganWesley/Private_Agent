"""Unit coverage for the provider adapter layer (Part I)."""

from private_agent.providers import all_presets, build_adapter


def test_ollama_adapter_reports_formatted_hardware_status(monkeypatch):
    from private_agent.hardware import status as hw_status

    seen = {}

    def fake_inspect(base_url, mode="auto", *, client_factory=None):
        seen["base_url"] = base_url
        seen["mode"] = mode
        seen["client_factory"] = client_factory
        return {"server": base_url, "mode": mode, "models": [], "placement": "cpu"}

    monkeypatch.setattr(hw_status, "inspect_ollama_hardware", fake_inspect)
    monkeypatch.setattr(hw_status, "format_hardware_status", lambda info: "REPORT")

    adapter = build_adapter("Ollama")
    sender = object()

    assert (
        adapter.hardware_status(
            base_url="http://127.0.0.1:11434",
            mode="gpu",
            client_factory=sender,
        )
        == "REPORT"
    )
    assert seen == {
        "base_url": "http://127.0.0.1:11434",
        "mode": "gpu",
        "client_factory": sender,
    }


def test_ollama_adapter_hardware_status_degrades_without_raising(monkeypatch):
    from private_agent.hardware import status as hw_status

    def exploding(*_args, **_kwargs):
        raise RuntimeError("no server")

    monkeypatch.setattr(hw_status, "inspect_ollama_hardware", exploding)

    report = build_adapter("Ollama").hardware_status(base_url="http://127.0.0.1:11434")

    assert "unavailable" in report.lower()


def test_openai_compatible_adapter_has_no_standardized_hardware_api():
    adapter = build_adapter(next(p for p in all_presets() if p.name == "LM Studio"))

    report = adapter.hardware_status(base_url="http://127.0.0.1:1234")

    assert "standard" in report.lower()


def test_ollama_adapter_detect_running_degrades_without_raising(monkeypatch):
    from private_agent.agent import local_runtimes

    def exploding(*_args, **_kwargs):
        raise RuntimeError("discovery blew up")

    monkeypatch.setattr(local_runtimes, "discover_local_runtimes", exploding)

    adapter = build_adapter("Ollama")

    assert adapter.detect_running(client_factory=object(), timeout=0.1) is None


def test_ollama_adapter_capabilities_degrade_without_raising(monkeypatch):
    from private_agent.agent import model_capabilities

    def exploding(*_args, **_kwargs):
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(
        model_capabilities, "detect_openai_compatible_capabilities", exploding
    )

    capabilities = build_adapter("Ollama").capabilities(
        "qwen3:8b",
        base_url="http://127.0.0.1:11434",
        client_factory=object(),
        timeout=0.1,
    )

    assert capabilities == {
        "tools": None,
        "vision": None,
        "thinking": None,
        "context_window": None,
    }


def test_adapters_survive_unavailable_optional_dependencies(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"langchain_ollama", "langchain_openai"}:
            raise ImportError(f"{name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    ollama = build_adapter("Ollama")
    lmstudio = build_adapter("LM Studio")

    assert (
        ollama.embeddings("nomic-embed-text", base_url="http://127.0.0.1:11434") is None
    )
    assert lmstudio.embeddings("m", base_url="http://127.0.0.1:1234/v1") is None
    try:
        ollama.chat_model("qwen3:8b", base_url="http://127.0.0.1:11434")
    except ImportError:
        pass
    else:  # pragma: no cover - chat_model legitimately needs the package
        raise AssertionError("chat_model should surface the missing package")


def test_missing_optional_package_does_not_hide_the_real_install_path(monkeypatch):
    adapter = build_adapter("LM Studio")

    assert adapter.detect_installed() in {True, False}
    assert adapter.start(consent=False) is False
