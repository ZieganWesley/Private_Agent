"""Built-in web search, fetch, and download tools."""

import contextlib
import os
import tempfile

import httpx
from langchain_core.tools import tool

from ..config import (
    MAX_DOWNLOAD_BYTES,
    MAX_FETCHED_WEBPAGE_CHARS,
    MAX_WEBPAGE_BYTES,
    NETWORK_REQUEST_TIMEOUT,
)
from ..sandbox import SandboxManager
from . import has_internet_connection, require_internet
from .network import (
    _get_limited_response_sync,
    public_only_sync_client,
)
from .schemas import DownloadWebFileInput, FetchWebpageInput, WebSearchInput
from .search import search_web


@tool(args_schema=WebSearchInput)
@require_internet
def web_search(query: str) -> str:
    """Search the web for up-to-date info, technical documentation, or code references."""
    try:
        return search_web(query)
    except (httpx.HTTPError, ValueError, OSError) as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"


@tool(args_schema=FetchWebpageInput)
@require_internet
def fetch_webpage(url: str, offset: int = 0, length: int | None = None) -> str:
    """Fetch, clean, and read article or documentation text from a URL."""

    try:
        from bs4 import BeautifulSoup

        with public_only_sync_client(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=NETWORK_REQUEST_TIMEOUT,
            track=False,
        ) as client:
            body, encoding = _get_limited_response_sync(
                client,
                url,
                MAX_WEBPAGE_BYTES,
            )
        soup = BeautifulSoup(body.decode(encoding, errors="replace"), "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        text = soup.get_text(separator="\n", strip=True)
        start = max(0, offset)
        if length is not None:
            end = start + length
        else:
            end = start + MAX_FETCHED_WEBPAGE_CHARS
        chunk = text[start:end]
        if end < len(text):
            chunk += "\n[Truncated...]"
        return chunk
    except ImportError:
        return "Failed to fetch webpage: Required package 'httpx' or 'beautifulsoup4' is not installed."
    except (httpx.HTTPError, ValueError, OSError) as exc:
        return f"Failed to fetch webpage: {exc}"


@tool(args_schema=DownloadWebFileInput)
def download_web_file(url: str, save_path: str) -> str:
    """Download a file securely inside the workspace."""
    try:
        target = SandboxManager.validate_path(save_path)
    except (ValueError, OSError) as exc:
        return f"Failed to download file: {exc}"
    if target.suffix.lower() in {".sh", ".exe", ".bat", ".cmd", ".msi"}:
        return (
            f"Error: Downloading executable/script extension '{target.suffix}' "
            "is restricted by security policy."
        )
    if not has_internet_connection():
        return "Error: Network unavailable. Please check your internet connection and try again."

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with public_only_sync_client(
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"},
                timeout=NETWORK_REQUEST_TIMEOUT,
                track=False,
            ) as client:
                body, _ = _get_limited_response_sync(
                    client,
                    url,
                    MAX_DOWNLOAD_BYTES,
                )
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as output:
                temporary_path = output.name
                output.write(body)
            os.replace(temporary_path, target)
        except (ValueError, OSError):
            if temporary_path:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temporary_path)
            raise
        return f"Success: File downloaded and saved to '{save_path}'."
    except (httpx.HTTPError, ValueError, OSError) as exc:
        return f"Failed to download file: {exc}"
