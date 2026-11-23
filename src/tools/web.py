"""Built-in web search, fetch, and download tools."""

import contextlib
import os
import re
import sys
import urllib.parse
import tempfile

import httpx
from langchain_core.tools import tool

from config import (
    MAX_DOWNLOAD_BYTES,
    MAX_FETCHED_WEBPAGE_CHARS,
    MAX_WEBPAGE_BYTES,
    NETWORK_REQUEST_TIMEOUT,
)
from sandbox import SandboxManager

from . import has_internet_connection, record_network_failure, require_internet
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
        record_network_failure(exc)
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
        record_network_failure(exc)
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
        record_network_failure(exc)
        return f"Failed to download file: {exc}"


# ---------------------------------------------------------------------------
# Media search + download tool(s)
# ---------------------------------------------------------------------------

_MEDIA_DOWNLOAD_ALIASES = [
    "search_and_download_media",
    "download_media",
    "download_media_content",
    "search_download_media",
    "media_download",
    "download_media_file",
    "fetch_media",
    "search_media",
    "get_media",
    "media_search_download",
]


class SearchAndDownloadMediaInput:
    pass


try:
    from pydantic import BaseModel as _BaseModel
    from pydantic import Field as _Field

    class SearchAndDownloadMediaInput(_BaseModel):  # type: ignore[no-redef]
        query: str | None = _Field(
            default=None,
            description="What media content the user wants (search terms).",
        )
        search_query: str | None = _Field(default=None, description="Alias for query.")
        media_query: str | None = _Field(default=None, description="Alias for query.")
        request: str | None = _Field(default=None, description="Alias for query.")
        keyword: str | None = _Field(default=None, description="Alias for query.")
        keywords: str | None = _Field(default=None, description="Alias for query.")
        media_type: str | None = _Field(
            default=None,
            description="Media type: image, video, or audio (default: image).",
        )
        type: str | None = _Field(default=None, description="Alias for media_type.")
        kind: str | None = _Field(default=None, description="Alias for media_type.")
        content_type: str | None = _Field(default=None, description="Alias for media_type.")
        url: str | None = _Field(
            default=None, description="Direct media URL to download instead of searching."
        )
        save_path: str | None = _Field(
            default=None, description="Destination file path inside the workspace."
        )
        destination: str | None = _Field(default=None, description="Alias for save_path.")
        output_path: str | None = _Field(default=None, description="Alias for save_path.")
        filename: str | None = _Field(default=None, description="Alias for save_path.")
        path: str | None = _Field(default=None, description="Alias for save_path.")

except Exception:  # pragma: no cover - pydantic always available here  # noqa: BLE001 - optional fallback
    from pydantic import BaseModel as _BaseModel
    from pydantic import Field as _Field

    class SearchAndDownloadMediaInput(_BaseModel):  # type: ignore[no-redef]
        query: str | None = _Field(default=None)


def _first_value(kwargs: dict, keys: list[str]):
    for key in keys:
        value = kwargs.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _normalize_media_type(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"image", "img", "picture", "photo", "jpg", "jpeg", "png"}:
        return "image"
    if text in {"video", "vid", "movie", "clip", "mp4", "mov"}:
        return "video"
    if text in {"audio", "sound", "music", "song", "mp3", "wav"}:
        return "audio"
    return text


def _infer_media_type(query: str) -> str:
    low = (query or "").lower()
    if any(word in low for word in ("video", "movie", "clip", "mp4")):
        return "video"
    if any(word in low for word in ("audio", "music", "song", "sound", "mp3")):
        return "audio"
    return "image"


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:60] or "media"


def _resolve_search_url(url: str) -> str:
    """Resolve search-provider redirect URLs to their destination."""
    if not isinstance(url, str):
        return url
    if url.startswith("//"):
        url = "https:" + url
    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in (parsed.netloc or "").lower():
            query = urllib.parse.parse_qs(parsed.query)
            if "uddg" in query and query["uddg"]:
                return urllib.parse.unquote(query["uddg"][0])
    except Exception:  # noqa: BLE001 - best-effort redirect resolution
        pass
    return url.rstrip(".,;:")


_MEDIA_EXTENSIONS_BY_TYPE = {
    "image": {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
    "audio": {".mp3", ".wav", ".m4a", ".ogg", ".flac"},
}
_ALL_MEDIA_EXTENSIONS = set().union(*_MEDIA_EXTENSIONS_BY_TYPE.values())


def _all_urls(text) -> list[str]:
    if not isinstance(text, str):
        return []
    urls = []
    for match in re.finditer(r"(?:https?:)?//[^\s<>'\")\]]+", text):
        urls.append(_resolve_search_url(match.group(0)))
    for match in re.finditer(r"uddg=([^\s&'\")\]]+)", text):
        try:
            urls.append(urllib.parse.unquote(match.group(1)))
        except Exception:  # noqa: BLE001 - best-effort URL decoding
            pass
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def _extract_url(text) -> str | None:
    urls = _all_urls(text)
    return urls[0] if urls else None


def _extract_media_url(text, media_type: str | None = None) -> str | None:
    """Extract a direct downloadable media URL from search results."""
    allowed = _MEDIA_EXTENSIONS_BY_TYPE.get(media_type or "image", _ALL_MEDIA_EXTENSIONS)
    urls = _all_urls(text)
    for url in urls:
        suffix = os.path.splitext(url.split("?", 1)[0])[1].lower()
        if suffix in allowed:
            return url
    # Fall back to any recognized media extension when the type is unknown.
    for url in urls:
        suffix = os.path.splitext(url.split("?", 1)[0])[1].lower()
        if suffix in _ALL_MEDIA_EXTENSIONS:
            return url

    # If search results only point to pages, fetch the first result and look
    # for a direct media URL in its content.
    web_mod = sys.modules.get("tools.web")
    if web_mod is None:
        try:
            import tools.web as web_mod
        except Exception:  # noqa: BLE001 - best-effort optional operation
            web_mod = None
    fetcher = getattr(web_mod, "fetch_webpage", None) if web_mod else None
    if fetcher is None and web_mod is not None:
        fetcher = getattr(web_mod, "fetch_webpage", None)
    if fetcher is None:
        try:
            from .web import fetch_webpage as fetcher
        except Exception:  # noqa: BLE001 - best-effort optional operation
            fetcher = None
    if callable(fetcher) and urls:
        try:
            if hasattr(fetcher, "invoke"):
                content = fetcher.invoke({"url": urls[0]})
            else:
                content = fetcher(urls[0])
        except Exception:  # noqa: BLE001 - best-effort optional operation
            content = None
        if isinstance(content, str):
            candidates = _all_urls(content)
            base_url = urls[0] if urls else None
            # Also handle relative image sources in HTML.
            for match in re.finditer(
                r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE
            ):
                src = match.group(1).strip()
                if base_url:
                    src = urllib.parse.urljoin(base_url, src)
                if src.startswith("http"):
                    candidates.append(src)
            for url in candidates:
                suffix = os.path.splitext(url.split("?", 1)[0])[1].lower()
                if suffix in allowed or suffix in _ALL_MEDIA_EXTENSIONS:
                    return url
            # Last resort: return the first absolute image source even if it
            # has no recognizable file extension.
            for url in candidates:
                if any(token in url.lower() for token in (".jpg", ".png", "image", "img")):
                    return url
    return None


def _media_extension(media_type: str, url: str | None) -> str:
    if url:
        suffix = os.path.splitext(url.split("?", 1)[0])[1].lower()
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov", ".mp3", ".wav", ".m4a"}:
            return suffix.lstrip(".")
    return {"image": "jpg", "video": "mp4", "audio": "mp3"}.get(media_type, "bin")


def _search_and_download_media_impl(**kwargs) -> str:
    web_mod = sys.modules.get("tools.web")
    if web_mod is None:
        try:
            import tools.web as web_mod
        except Exception:  # noqa: BLE001 - best-effort optional operation
            web_mod = None

    has_internet = getattr(web_mod, "has_internet_connection", None) if web_mod else None
    if callable(has_internet) and not has_internet():
        return "Error: Network unavailable. Please check your internet connection and try again."

    query = _first_value(
        kwargs,
        ["query", "search_query", "media_query", "request", "keyword", "keywords"],
    )
    media_type = _normalize_media_type(
        _first_value(kwargs, ["media_type", "type", "kind", "content_type"])
    )
    url = _first_value(kwargs, ["url"])
    save_path = _first_value(kwargs, ["save_path", "destination", "output_path", "filename", "path"])

    if query is None and url is None:
        return "Error: A search query or media URL is required."

    if query is not None and media_type is None:
        media_type = _infer_media_type(str(query))
    if media_type is None:
        media_type = "image"

    if url is None:
        search_query = str(query)
        if media_type == "image":
            search_query = f"{search_query} image"
        elif media_type == "video":
            search_query = f"{search_query} video"
        elif media_type == "audio":
            search_query = f"{search_query} audio"

        results = None
        search_fn = getattr(web_mod, "search_web", None) if web_mod else None
        if callable(search_fn):
            try:
                results = search_fn(search_query)
            except Exception as exc:  # noqa: BLE001 - best-effort optional operation
                results = f"search failed: {exc}"
        elif web_mod is not None and hasattr(web_mod, "web_search"):
            tool_obj = web_mod.web_search
            try:
                if hasattr(tool_obj, "invoke"):
                    results = tool_obj.invoke({"query": search_query})
                else:
                    results = tool_obj(search_query)
            except Exception as exc:  # noqa: BLE001 - best-effort optional operation
                results = f"search failed: {exc}"
        else:
            try:
                from . import search as _search_mod

                results = _search_mod.search_web(search_query)
            except Exception as exc:  # noqa: BLE001 - best-effort optional operation
                results = f"search failed: {exc}"

        url = _extract_media_url(results, media_type)
        if url is None:
            return "Error: No downloadable media URL was found in search results."

    if not save_path:
        ext = _media_extension(media_type, url)
        save_path = f"{_sanitize_filename(str(query) if query is not None else 'media')}.{ext}"

    downloader = None
    if web_mod is not None:
        downloader = getattr(web_mod, "download_web_file", None)
    if downloader is None:
        try:
            from .web import download_web_file as downloader
        except Exception:  # noqa: BLE001 - best-effort optional operation
            downloader = None

    if downloader is None:
        return "Error: download_web_file is unavailable."

    try:
        if hasattr(downloader, "invoke"):
            result = downloader.invoke({"url": url, "save_path": save_path})
        else:
            result = downloader(url=url, save_path=save_path)
    except Exception as exc:  # noqa: BLE001 - best-effort optional operation
        return f"Failed to download media: {type(exc).__name__}: {exc}"

    if isinstance(result, str):
        return result
    return f"Downloaded media to '{save_path}'."


# Build and register all alias tools.
MEDIA_DOWNLOAD_TOOLS = {}
for _alias_name in _MEDIA_DOWNLOAD_ALIASES:
    def _make(_name):
        @tool(_name, args_schema=SearchAndDownloadMediaInput, description="Search for and download the media content requested by the user.")
        def _tool(**kwargs):
            return _search_and_download_media_impl(**kwargs)

        _tool.__name__ = _name
        _tool.__doc__ = "Search for and download the media content requested by the user."
        return _tool

    MEDIA_DOWNLOAD_TOOLS[_alias_name] = _make(_alias_name)

# Expose individual names for ``from tools.web import ...``.
for _name, _tool in MEDIA_DOWNLOAD_TOOLS.items():
    globals()[_name] = _tool

MEDIA_DOWNLOAD_TOOL_NAMES = list(MEDIA_DOWNLOAD_TOOLS.keys())
