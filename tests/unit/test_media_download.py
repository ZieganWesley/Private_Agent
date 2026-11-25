"""Media search-and-download tool tests."""

import types

import pytest

from tools import download_media


@pytest.fixture
def fake_web(monkeypatch, temp_workspace):
    import tools.web as web

    monkeypatch.setattr(web, "has_internet_connection", lambda: True, raising=False)

    def search(query):
        return "Title: cat photo\nURL: https://example.com/cat.jpg"

    monkeypatch.setattr(web, "search_web", search, raising=False)

    def download(url=None, save_path=None, **kwargs):
        target = temp_workspace / save_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("media bytes", encoding="utf-8")
        return f"Success: File downloaded and saved to '{save_path}'."

    fake_downloader = types.SimpleNamespace(invoke=lambda args: download(**args))
    monkeypatch.setattr(web, "download_web_file", fake_downloader, raising=False)
    return temp_workspace


def test_media_download_searches_and_saves_file(fake_web):
    result = download_media.invoke({"query": "cat photo"})

    assert "Success" in result
    assert "cat_photo.jpg" in result
    assert (fake_web / "cat_photo.jpg").is_file()
    assert (fake_web / "cat_photo.jpg").read_text(encoding="utf-8") == "media bytes"


def test_media_download_reports_missing_search_result(monkeypatch, temp_workspace):
    import tools.web as web

    monkeypatch.setattr(web, "has_internet_connection", lambda: True, raising=False)
    monkeypatch.setattr(web, "search_web", lambda query: "No results", raising=False)

    result = download_media.invoke({"query": "nothing"})

    assert "No downloadable media URL" in result
    assert not any(temp_workspace.iterdir())


def test_media_download_uses_direct_url_and_custom_save_path(
    monkeypatch, temp_workspace
):
    import tools.web as web

    monkeypatch.setattr(web, "has_internet_connection", lambda: True, raising=False)

    def download(url=None, save_path=None, **kwargs):
        target = temp_workspace / save_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("direct bytes", encoding="utf-8")
        return f"Success: File downloaded and saved to '{save_path}'."

    monkeypatch.setattr(
        web,
        "download_web_file",
        types.SimpleNamespace(invoke=lambda args: download(**args)),
        raising=False,
    )

    result = download_media.invoke(
        {
            "query": "ignored",
            "url": "https://example.com/direct.png",
            "save_path": "nested/out.png",
        }
    )

    assert "Success" in result
    assert (temp_workspace / "nested" / "out.png").is_file()


def test_media_download_offline_returns_error(monkeypatch, temp_workspace):
    import tools.web as web

    monkeypatch.setattr(web, "has_internet_connection", lambda: False, raising=False)

    result = download_media.invoke({"query": "cat"})

    assert "Network unavailable" in result
