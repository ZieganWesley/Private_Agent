"""Configurable web search providers, with an explicit scraper fallback."""

import logging

import httpx

from .. import config
from ..utils.run_logging import RUN_LOGGER, log_event
from .network import search_duckduckgo, searxng_search

PROVIDERS = ("auto", "searxng", "duckduckgo")

# Transport, configuration and parsing failures a provider may raise. httpx
# errors derive from Exception rather than OSError, so they are listed here.
_SEARCH_FAILURES = (httpx.HTTPError, ValueError, OSError)

FALLBACK_NOTICE = (
    "Warning: your SearXNG instance failed ({error}), so these results come "
    "from DuckDuckGo scraping instead. Set [network] search_fallback = false "
    "to fail rather than fall back."
)


def search_web(
    query: str,
    *,
    provider: str | None = None,
    searxng_url: str | None = None,
    fallback: bool | None = None,
) -> str:
    """Run the configured search provider and return formatted results.

    ``auto`` uses a configured SearXNG instance and otherwise scrapes
    DuckDuckGo, so a default install needs no extra service. When SearXNG is
    the chosen provider and it fails, the scraper answers only if
    ``[network] search_fallback`` allows it: the returned text then starts with
    an explicit notice, and with the fallback disabled the failure is raised
    instead, so privacy is never downgraded without the user being told.
    """
    chosen = str(config.SEARCH_PROVIDER if provider is None else provider)
    chosen = chosen.strip().lower()
    if chosen not in PROVIDERS:
        log_event(
            RUN_LOGGER,
            "search.unknown_provider",
            level=logging.WARNING,
            provider=chosen,
            fallback="auto",
        )
        chosen = "auto"

    url = str(config.SEARXNG_URL if searxng_url is None else searxng_url).strip()
    if chosen == "auto":
        chosen = "searxng" if url else "duckduckgo"
    if chosen == "duckduckgo":
        return search_duckduckgo(query)
    if not url:
        raise ValueError(
            "Web search provider 'searxng' requires network.searxng_url to be set."
        )

    allow_fallback = bool(config.SEARCH_FALLBACK if fallback is None else fallback)

    try:
        return searxng_search(query, base_url=url)
    except _SEARCH_FAILURES as exc:
        log_event(
            RUN_LOGGER,
            "search.searxng_failed",
            level=logging.WARNING,
            error_type=type(exc).__name__,
            fallback="duckduckgo" if allow_fallback else "disabled",
            exc_info=True,
        )
        if not allow_fallback:
            raise ValueError(
                "SearXNG search failed and the scraper fallback is disabled "
                f"([network] search_fallback = false): {type(exc).__name__}: {exc}"
            ) from exc
        results = search_duckduckgo(query)
        return f"{FALLBACK_NOTICE.format(error=type(exc).__name__)}\n\n{results}"
