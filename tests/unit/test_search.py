"""Web search provider selection, SearXNG parsing and scraper fallback."""

import inspect
import json

import httpx
import pytest

from tools import network, search, web


class _FakeResponse:
    def __init__(self, body, encoding="utf-8"):
        self._body = body
        self.encoding = encoding
        self.headers = {}
        self.is_redirect = False

    def raise_for_status(self):
        return None

    def iter_bytes(self):
        yield self._body


class _FakeClient:
    def __init__(self, body, calls):
        self._body = body
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def stream(self, method, url):
        self.calls.append((method, url))
        body = self._body

        class _Context:
            def __enter__(self_inner):
                return _FakeResponse(body)

            def __exit__(self_inner, *exc_info):
                return False

        return _Context()


def _client_factory(body, calls):
    def factory(**_kwargs):
        return _FakeClient(body, calls)

    return factory


def test_searxng_search_formats_json_results_like_the_scraper():
    calls = []
    payload = {
        "results": [
            {
                "title": "Asyncio basics",
                "content": "Event loops.",
                "url": "https://a.example",
            },
            {"title": "Missing URL", "content": "skipped"},
            {"title": "Second", "content": "Task groups.", "url": "https://b.example"},
        ]
    }

    result = network.searxng_search(
        "asyncio",
        base_url="http://127.0.0.1:8080",
        client_factory=_client_factory(json.dumps(payload).encode(), calls),
    )

    assert result == (
        "Title: Asyncio basics\nSnippet: Event loops.\nURL: https://a.example"
        "\n---\n"
        "Title: Second\nSnippet: Task groups.\nURL: https://b.example"
    )
    method, url = calls[0]
    assert method == "GET"
    assert url.startswith("http://127.0.0.1:8080/search?")
    assert "format=json" in url
    assert "q=asyncio" in url


def test_searxng_search_reports_no_results():
    calls = []
    result = network.searxng_search(
        "nothing",
        base_url="http://127.0.0.1:8080",
        client_factory=_client_factory(b'{"results": []}', calls),
    )

    assert result == "No web search results found."


def test_searxng_search_caps_results_and_ignores_malformed_entries():
    calls = []
    payload = {
        "results": [
            {
                "title": f"Result {index}",
                "content": "",
                "url": f"https://{index}.example",
            }
            for index in range(10)
        ]
        + ["not a mapping"]
    }

    result = network.searxng_search(
        "many",
        base_url="http://127.0.0.1:8080/",
        client_factory=_client_factory(json.dumps(payload).encode(), calls),
    )

    assert result.count("Title:") == network.MAX_SEARCH_RESULTS


def test_searxng_search_requires_a_configured_url():
    with pytest.raises(ValueError, match="not configured"):
        network.searxng_search("q", base_url="  ")


def test_searxng_search_rejects_a_non_http_endpoint():
    with pytest.raises(ValueError):
        network.searxng_search("q", base_url="file:///tmp/searxng")


def test_searxng_search_raises_value_error_when_the_instance_returns_html():
    """HTML (JSON API disabled) must surface as ValueError so callers fall back."""
    calls = []
    with pytest.raises(ValueError):
        network.searxng_search(
            "q",
            base_url="http://127.0.0.1:8080",
            client_factory=_client_factory(b"<html>not json</html>", calls),
        )


@pytest.mark.parametrize(
    ("provider", "url", "expected"),
    [
        ("auto", "", "duckduckgo"),
        ("auto", "http://127.0.0.1:8080", "searxng"),
        ("duckduckgo", "http://127.0.0.1:8080", "duckduckgo"),
        ("searxng", "http://127.0.0.1:8080", "searxng"),
        ("SEARXNG", " http://127.0.0.1:8080 ", "searxng"),
    ],
)
def test_search_web_selects_the_configured_provider(
    monkeypatch, provider, url, expected
):
    used = []

    def fake_searxng(query, *, base_url):
        used.append("searxng")
        return f"searxng:{query}"

    def fake_duckduckgo(query):
        used.append("duckduckgo")
        return f"duckduckgo:{query}"

    monkeypatch.setattr(search, "searxng_search", fake_searxng)
    monkeypatch.setattr(search, "search_duckduckgo", fake_duckduckgo)

    result = search.search_web("query", provider=provider, searxng_url=url)

    assert used == [expected]
    assert result == f"{expected}:query"


def test_unknown_provider_is_treated_as_auto(monkeypatch):
    used = []
    monkeypatch.setattr(
        search,
        "search_duckduckgo",
        lambda query: used.append("duckduckgo") or "scraped",
    )

    assert search.search_web("q", provider="bing", searxng_url="") == "scraped"
    assert used == ["duckduckgo"]


def test_explicit_searxng_without_a_url_is_a_configuration_error():
    with pytest.raises(ValueError, match="searxng_url"):
        search.search_web("q", provider="searxng", searxng_url="")


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection refused"),
        ValueError("HTML instead of JSON"),
        OSError("network dropped"),
    ],
)
def test_a_failing_searxng_falls_back_with_an_explicit_notice(monkeypatch, failure):
    def failing_searxng(query, *, base_url):
        raise failure

    monkeypatch.setattr(search, "searxng_search", failing_searxng)
    monkeypatch.setattr(search, "search_duckduckgo", lambda query: f"scraped:{query}")

    result = search.search_web(
        "q",
        provider="searxng",
        searxng_url="http://127.0.0.1:8080",
    )

    notice, results = result.split("\n\n")
    assert results == "scraped:q"
    assert notice.startswith("Warning: your SearXNG instance failed (")
    assert type(failure).__name__ in notice
    assert "DuckDuckGo scraping" in notice
    assert "[network] search_fallback = false" in notice


def test_the_fallback_can_be_disabled(monkeypatch):
    def failing_searxng(query, *, base_url):
        raise httpx.ConnectError("connection refused")

    scraped = []
    monkeypatch.setattr(search, "searxng_search", failing_searxng)
    monkeypatch.setattr(
        search, "search_duckduckgo", lambda query: scraped.append(query) or "scraped"
    )

    with pytest.raises(ValueError, match="scraper fallback is disabled"):
        search.search_web(
            "q",
            provider="searxng",
            searxng_url="http://127.0.0.1:8080",
            fallback=False,
        )

    assert scraped == []


def test_the_fallback_setting_applies_when_no_override_is_passed(monkeypatch):
    def failing_searxng(query, *, base_url):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(search, "searxng_search", failing_searxng)
    monkeypatch.setattr(search, "search_duckduckgo", lambda query: f"scraped:{query}")
    monkeypatch.setattr(search.config, "SEARCH_FALLBACK", False)

    with pytest.raises(ValueError, match="scraper fallback is disabled"):
        search.search_web("q", provider="searxng", searxng_url="http://127.0.0.1:8080")

    monkeypatch.setattr(search.config, "SEARCH_FALLBACK", True)
    assert search.search_web(
        "q", provider="searxng", searxng_url="http://127.0.0.1:8080"
    ).endswith("\n\nscraped:q")


def test_the_notice_is_not_added_when_searxng_answers(monkeypatch):
    monkeypatch.setattr(search, "searxng_search", lambda query, *, base_url: "clean")
    monkeypatch.setattr(search, "search_duckduckgo", lambda query: "scraped")

    assert (
        search.search_web("q", provider="searxng", searxng_url="http://127.0.0.1:8080")
        == "clean"
    )


def test_web_search_tool_uses_the_provider_layer(monkeypatch):
    monkeypatch.setattr(web, "has_internet_connection", lambda: True)
    monkeypatch.setattr(web, "search_web", lambda query: f"result:{query}")

    assert web.web_search.invoke({"query": "python"}) == "result:python"


def test_web_search_tool_shows_the_fallback_notice_and_refuses_it_when_disabled(
    monkeypatch,
):
    def failing_searxng(query, *, base_url):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(web, "has_internet_connection", lambda: True)
    monkeypatch.setattr(search, "searxng_search", failing_searxng)
    monkeypatch.setattr(search, "search_duckduckgo", lambda query: f"scraped:{query}")
    monkeypatch.setattr(search.config, "SEARCH_PROVIDER", "searxng")
    monkeypatch.setattr(search.config, "SEARXNG_URL", "http://127.0.0.1:8080")

    assert web.web_search.invoke({"query": "python"}).startswith(
        "Warning: your SearXNG"
    )

    monkeypatch.setattr(search.config, "SEARCH_FALLBACK", False)
    refused = web.web_search.invoke({"query": "python"})
    assert refused.startswith("Web search failed: ValueError:")
    assert "scraper fallback is disabled" in refused


def test_web_search_tool_reports_configuration_errors(monkeypatch):
    def failing_search(_query):
        raise ValueError("provider misconfigured")

    monkeypatch.setattr(web, "has_internet_connection", lambda: True)
    monkeypatch.setattr(web, "search_web", failing_search)

    result = web.web_search.invoke({"query": "python"})

    assert result == "Web search failed: ValueError: provider misconfigured"


def test_the_relaxed_validator_is_opt_in_only():
    """Model- or page-supplied URLs must keep the public-only DNS policy."""
    parameters = inspect.signature(network._get_limited_response_sync).parameters

    assert parameters["validate"].default is network.validate_outbound_url
