"""Configurable web search providers, with the scraper as a fallback."""

import logging

import httpx

from .. import config
from ..run_logging import RUN_LOGGER, log_event
from .network import search_duckduckgo, searxng_search

PROVIDERS = ("auto", "searxng", "duckduckgo")

# Transport, configuration and parsing failures a provider may raise. httpx
# errors derive from Exception rather than OSError, so they are listed here.
_SEARCH_FAILURES = (httpx.HTTPError, ValueError, OSError)


def search_web(
    query: str,
    *,
    provider: str | None = None,
    searxng_url: str | None = None,
) -> str:
    """Run the configured search provider and return formatted results.

    ``auto`` uses a configured SearXNG instance and otherwise scrapes
    DuckDuckGo, so a default install needs no extra service. An explicitly
    chosen SearXNG that fails falls back to the scraper and logs why, so a
    transient outage does not remove web search for the rest of the session.
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

    try:
        return searxng_search(query, base_url=url)
    except _SEARCH_FAILURES as exc:
        log_event(
            RUN_LOGGER,
            "search.searxng_failed",
            level=logging.WARNING,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        return search_duckduckgo(query)
