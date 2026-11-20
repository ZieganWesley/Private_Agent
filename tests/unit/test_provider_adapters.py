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
