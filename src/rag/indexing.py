import hashlib
import json
import os
import pathlib
import shutil
import sqlite3
import stat as stat_module
import tempfile
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import quote

from langchain_core.documents import Document

from config import (
    APP_CONFIG,
    CHUNKING_MIN_CHUNK_CHARS,
    CHUNKING_OVERLAP,
    CHUNKING_SEMANTIC_BUFFER,
    CHUNKING_SEMANTIC_MAX_SENTENCES,
    CHUNKING_SEMANTIC_PERCENTILE,
    CHUNKING_SEPARATORS,
    CHUNKING_SIZE,
    CHUNKING_STRATEGY,
    CHUNKING_UNIT,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    HARDWARE_ACCELERATION_MODE,
    OLLAMA_BASE_URL,
    RAG_EXCLUDED_FILE_TYPES,
    RAG_FILE_TYPES,
    RAG_INDEX_PATH,
    RAG_MAX_CORPUS_BYTES,
    RAG_MAX_DOCUMENTS,
    RAG_MAX_FILE_BYTES,
    RAG_MAX_PDF_PAGES,
    RAG_SQLITE_MAX_CELL_CHARS,
    RAG_SQLITE_MAX_ROWS_PER_TABLE,
    RAG_SQLITE_ROWS_PER_CHUNK,
)
from hardware import ollama_acceleration_options
from ui.console import console

from .chunking import ChunkingSettings, split_text

SQLITE_EXTENSIONS = frozenset({".db", ".sqlite", ".sqlite3", ".db3"})
SUPPORTED_EXTENSIONS = (
    frozenset(RAG_FILE_TYPES) | SQLITE_EXTENSIONS
) - frozenset(RAG_EXCLUDED_FILE_TYPES)
_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_ROWS_PER_CHUNK = RAG_SQLITE_ROWS_PER_CHUNK
_SQLITE_MAX_CELL_CHARS = RAG_SQLITE_MAX_CELL_CHARS


def _iter_source_files(root: pathlib.Path):
    """Yield (path, stat) for supported files under root.

    Symbolic links to files and directories are followed (targets may live
    outside root). Hard links and symlinks that reach the same file are indexed
    once, and directory cycles are not re-entered.
    """
    seen_directories = set()
    seen_files = set()
    for directory, subdirectories, filenames in os.walk(root, followlinks=True):
        real_directory = os.path.realpath(directory)
        if real_directory in seen_directories:
            subdirectories[:] = []
            continue
        seen_directories.add(real_directory)
        subdirectories[:] = sorted(
            name for name in subdirectories if not name.startswith(".")
        )
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if pathlib.Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            file_path = pathlib.Path(directory) / name
            try:
                file_stat = file_path.stat()
            except OSError:
                continue
            if not stat_module.S_ISREG(file_stat.st_mode):
                continue
            identity = (file_stat.st_dev, file_stat.st_ino)
            if identity in seen_files:
                continue
            seen_files.add(identity)
            yield file_path, file_stat


def embedding_signature(
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
) -> str:
    """Identify the embedder that produced an index (provider, model, endpoint)."""
    provider = (
        embedding_provider if embedding_provider is not None else EMBEDDING_PROVIDER
    )
    model = embedding_model if embedding_model is not None else EMBEDDING_MODEL
    base_url = (
        embedding_base_url if embedding_base_url is not None else EMBEDDING_BASE_URL
    )
    if provider == "ollama":
        return f"ollama:{model}"
    if provider == "none":
        return "none"
    return f"{provider}:{model}@{base_url.rstrip('/')}"


def _source_fingerprint(file_path: pathlib.Path, file_stat) -> dict:
    size = file_stat.st_size
    mtime_ns = file_stat.st_mtime_ns
    if file_path.suffix.lower() in SQLITE_EXTENSIONS:
        # Writes to a WAL-mode database may live only in the -wal file.
        try:
            wal_stat = file_path.with_name(file_path.name + "-wal").stat()
            size += wal_stat.st_size
            mtime_ns = max(mtime_ns, wal_stat.st_mtime_ns)
        except OSError:
            pass
    return {"mtime_ns": mtime_ns, "size": size}


def knowledge_base_signature(docs_path: str | None) -> str | None:
    """Cheap digest of the indexable files' paths, sizes and mtimes."""
    if not docs_path or not str(docs_path).strip():
        return None
    root = pathlib.Path(docs_path).expanduser().resolve()
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path, file_stat in _iter_source_files(root):
        fingerprint = _source_fingerprint(file_path, file_stat)
        digest.update(
            f"{file_path}\0{fingerprint['mtime_ns']}\0{fingerprint['size']}\n".encode(
                "utf-8", "surrogateescape"
            )
        )
    return digest.hexdigest()


def _sqlite_cell(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (bytes, bytearray)):
        return f"<blob {len(value)} bytes>"
    text = str(value).replace("\n", " ")
    if len(text) > _SQLITE_MAX_CELL_CHARS:
        text = text[:_SQLITE_MAX_CELL_CHARS] + "..."
    return text


def _read_sqlite_pages(file_path: pathlib.Path) -> list:
    """Render each table's rows as text batches using a read-only connection."""
    with file_path.open("rb") as database_file:
        if database_file.read(len(_SQLITE_HEADER)) != _SQLITE_HEADER:
            raise ValueError("not a SQLite database file")
    connection = sqlite3.connect(
        f"file:{quote(str(file_path))}?mode=ro", uri=True, timeout=5
    )
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.text_factory = lambda raw: raw.decode("utf-8", "replace")
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        pages = []
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            cursor = connection.execute(
                f"SELECT * FROM {quoted} LIMIT ?",  # nosec B608 - identifier is quoted, limit is bound
                (RAG_SQLITE_MAX_ROWS_PER_TABLE,),
            )
            columns = [column[0] for column in cursor.description]
            batch = []
            rows_seen = 0
            for row in cursor:
                rows_seen += 1
                batch.append(
                    " | ".join(
                        f"{column}={_sqlite_cell(value)}"
                        for column, value in zip(columns, row, strict=False)
                    )
                )
                if len(batch) >= _SQLITE_ROWS_PER_CHUNK:
                    pages.append((f"SQLite table {table}\n" + "\n".join(batch), None))
                    batch = []
            if batch:
                pages.append((f"SQLite table {table}\n" + "\n".join(batch), None))
            if not rows_seen:
                pages.append(
                    (
                        f"SQLite table {table} (columns: {', '.join(columns)}; no rows)",
                        None,
                    )
                )
        return pages
    finally:
        connection.close()


def reset_knowledge_base(index_path: str | None = None) -> str:
    """Remove only the configured Chroma index directory after path safety checks."""
    configured_path = pathlib.Path(index_path or RAG_INDEX_PATH).expanduser()
    if configured_path.is_symlink():
        raise ValueError("Refusing to reset a symbolic-link Chroma index path.")
    target = configured_path.resolve()
    protected_paths = {
        pathlib.Path("/").resolve(),
        pathlib.Path.home().resolve(),
        pathlib.Path.cwd().resolve(),
    }
    if target in protected_paths:
        raise ValueError(f"Refusing to reset protected directory '{target}'.")
    if target.exists() and not target.is_dir():
        raise ValueError(f"Chroma index path '{target}' is not a directory.")
    if target.exists():
        shutil.rmtree(target)
    return str(target)


def load_chunking_settings() -> ChunkingSettings:
    """Build chunking settings from config, falling back to defaults when invalid."""
    try:
        return ChunkingSettings(
            strategy=CHUNKING_STRATEGY,
            chunk_size=CHUNKING_SIZE,
            chunk_overlap=CHUNKING_OVERLAP,
            unit=CHUNKING_UNIT
            if CHUNKING_UNIT in ("chars", "words", "tokens")
            else "chars",
            min_chunk_chars=CHUNKING_MIN_CHUNK_CHARS,
            separators=CHUNKING_SEPARATORS,
            semantic_breakpoint_percentile=CHUNKING_SEMANTIC_PERCENTILE,
            semantic_buffer_sentences=CHUNKING_SEMANTIC_BUFFER,
            semantic_max_sentences=CHUNKING_SEMANTIC_MAX_SENTENCES,
        )
    except ValueError as exc:
        console.print(
            f"[warning][Warning] Invalid chunking settings ({exc}); "
            "using defaults.[/warning]"
        )
        return ChunkingSettings(chunk_size=1200, chunk_overlap=200)


def _split_into_chunks(
    content: str,
    settings: ChunkingSettings | None = None,
    extension: str = "",
    embed=None,
) -> list[str]:
    return split_text(content, settings or load_chunking_settings(), extension, embed)


def _read_index_state(state_path: pathlib.Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"RAG index state at '{state_path}' is unreadable; preserve the index "
            "and repair or explicitly rebuild it before indexing."
        ) from exc
    if not isinstance(state, dict) or any(
        not isinstance(source, str)
        or not source
        or not pathlib.Path(source).is_absolute()
        or not isinstance(fingerprint, dict)
        or type(fingerprint.get("mtime_ns")) is not int
        or type(fingerprint.get("size")) is not int
        or fingerprint["mtime_ns"] < 0
        or fingerprint["size"] < 0
        for source, fingerprint in state.items()
    ):
        raise ValueError(
            f"RAG index state at '{state_path}' has an unsupported format; "
            "preserve the index and repair or explicitly rebuild it before indexing."
        )
    return state


def _write_index_state_atomic(state_path: pathlib.Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_path.parent,
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as state_file:
            temporary_path = pathlib.Path(state_file.name)
            json.dump(state, state_file, indent=4)
            state_file.flush()
            os.fsync(state_file.fileno())
        os.replace(temporary_path, state_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _preserve_corrupt_index(index_path: pathlib.Path) -> pathlib.Path:
    backup_path = index_path.with_name(
        f"{index_path.name}.backup-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex}"
    )
    os.replace(index_path, backup_path)
    index_path.mkdir(parents=True, exist_ok=False)
    return backup_path


def _ensure_rag_dependencies():
    global Chroma, OllamaEmbeddings
    if Chroma is None:
        try:
            from langchain_chroma import Chroma as _Chroma

            Chroma = _Chroma
        except ImportError:  # noqa: BLE001 - optional dependency
            Chroma = None
    if OllamaEmbeddings is None:
        try:
            from langchain_ollama import OllamaEmbeddings as _OllamaEmbeddings

            OllamaEmbeddings = _OllamaEmbeddings
        except Exception:  # noqa: BLE001 - optional dependency
            OllamaEmbeddings = None


Chroma = None
OllamaEmbeddings = None


_CHUNK_CACHE: dict[str, tuple[dict, list]] = {}


def clear_chunk_cache() -> None:
    """Drop cached parsed chunks so the next refresh re-parses every file."""
    _CHUNK_CACHE.clear()


def initialize_knowledge_base(
    docs_path: str | None,
    *,
    index_path: str | None = None,
    confirm_rebuild: Callable[[str], bool] | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_base_url: str | None = None,
):
    _ensure_rag_dependencies()
    """Initializes the Chroma vector store exclusively when an explicit, valid knowledge base folder path is provided."""
    _embedding_provider = (
        embedding_provider if embedding_provider is not None else EMBEDDING_PROVIDER
    )
    _embedding_model = (
        embedding_model if embedding_model is not None else EMBEDDING_MODEL
    )
    _embedding_base_url = (
        embedding_base_url if embedding_base_url is not None else EMBEDDING_BASE_URL
    )
    if not docs_path:
        return None
    if isinstance(docs_path, str):
        raw_paths = [
            part.strip()
            for part in docs_path.replace(";", ",").split(",")
            if part.strip()
        ]
    else:
        raw_paths = [str(part).strip() for part in docs_path if str(part).strip()]
    valid_paths = []
    for raw in raw_paths:
        candidate = pathlib.Path(raw).expanduser().resolve()
        if candidate.is_dir():
            valid_paths.append(candidate)
        else:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                valid_paths.append(candidate)
            except OSError:
                console.print(
                    f"[warning][Warning] Provided KB path '{candidate}' is not a "
                    f"valid directory. Skipping it.[/warning]"
                )
    if not valid_paths:
        return None

    chroma_persist_dir = os.path.abspath(index_path or RAG_INDEX_PATH)
    state_file_path = pathlib.Path(chroma_persist_dir) / ".index_state.json"
    os.makedirs(chroma_persist_dir, exist_ok=True)
    try:
        index_state = _read_index_state(state_file_path)
    except ValueError as exc:
        if confirm_rebuild is None or not confirm_rebuild(str(exc)):
            console.print(f"[error][RAG state error] {exc}[/error]")
            return None
        try:
            backup_path = _preserve_corrupt_index(state_file_path.parent)
        except OSError as recovery_error:
            console.print(
                f"[error][RAG recovery error] Could not preserve the damaged "
                f"index: {recovery_error}[/error]"
            )
            return None
        console.print(
            f"[warning][RAG recovery][/warning] Preserved the previous index at "
            f"'{backup_path}' and will build a new index."
        )
        index_state = {}

    embeddings = None
    signature = None
    if Chroma is None:
        console.print(
            "[warning][RAG] 'langchain-chroma' is not installed; RAG will use "
            "keyword search only. Install the 'rag' extra for semantic search.[/warning]"
        )
        _embedding_provider = "none"

    if _embedding_provider == "none":
        console.print(
            "[info][Info] Embedding provider disabled; RAG will use keyword search only.[/info]"
        )
    else:
        signature = embedding_signature(
            embedding_provider=_embedding_provider,
            embedding_model=_embedding_model,
            embedding_base_url=_embedding_base_url,
        )
        recorded_signatures = {
            str(fingerprint.get("embedding"))
            for fingerprint in index_state.values()
            if isinstance(fingerprint, dict) and fingerprint.get("embedding")
        }
        if recorded_signatures and signature not in recorded_signatures:
            console.print(
                "[warning][RAG] The embedding provider or model changed "
                f"({', '.join(sorted(recorded_signatures))} -> {signature}); "
                "rebuilding the index.[/warning]"
            )

        console.print(
            f"[info][Info] Initializing local embedding model '{_embedding_model}' via "
            f"{_embedding_provider}...[/info]"
        )
        try:
            if _embedding_provider == "ollama":
                from tools import (
                    ollama_langchain_client_kwargs,
                    track_ollama_http_clients,
                )

                embeddings = OllamaEmbeddings(
                    model=_embedding_model,
                    base_url=OLLAMA_BASE_URL,
                    **ollama_langchain_client_kwargs(),
                    **ollama_acceleration_options(HARDWARE_ACCELERATION_MODE),
                )
                track_ollama_http_clients(embeddings)
            else:
                from agent.local_runtimes import _is_loopback_url
                from providers.adapters import OpenAICompatibleAdapter

                if not _is_loopback_url(_embedding_base_url):
                    console.print(
                        f"[warning][Remote host][/warning] {_embedding_base_url} is not "
                        "on this machine; document chunks leave the local device."
                    )
                embeddings = OpenAICompatibleAdapter().embeddings(
                    _embedding_model, base_url=_embedding_base_url
                )
                if embeddings is None:
                    console.print(
                        "[error][Error] Could not create the embedding client for "
                        f"'{_embedding_base_url}'.[/error]"
                    )
                    return None
        except (ValueError, OSError, TypeError, RuntimeError) as e:  # noqa: BLE001 - optional dependency boundary
            console.print(f"[error][Error] Could not bind embedding model: {e}[/error]")
            return None

        # Verify the embedding runtime is actually reachable before indexing, so a
        # configured-but-absent runtime (for example Ollama on a llama.cpp-only
        # machine) disables RAG instead of failing on every later search.
        if hasattr(embeddings, "embed_query"):
            try:
                embeddings.embed_query("embedding availability check")
            except Exception as e:  # noqa: BLE001 - embedding runtimes are optional
                console.print(
                    f"[warning][RAG] Embedding runtime is unavailable "
                    f"({type(e).__name__}); continuing without RAG.[/warning]"
                )
                return None

    if Chroma is not None:
        console.print(
            f"[bold success][Chroma DB][/bold success] Initializing local project vector store at: [info]{chroma_persist_dir}[/info]"
        )

    db_exists = os.path.exists(chroma_persist_dir) and any(
        f for f in os.listdir(chroma_persist_dir) if f != ".index_state.json"
    )

    changed_docs = []
    all_parsed_docs = []
    updated_state = {}
    failed_sources = set()
    total_source_bytes = 0

    chunking = load_chunking_settings()

    _source_items = []
    for _collection_path in valid_paths:
        _collection_name = _collection_path.name or "default"
        for _file_path, _stat in _iter_source_files(_collection_path):
            _source_items.append((_file_path, _stat, _collection_name))

    for file_path, stat, collection_name in _source_items:
        ext = file_path.suffix.lower()
        try:
            file_key = str(file_path)
            if stat.st_size > RAG_MAX_FILE_BYTES:
                console.print(
                    f"[warning][Warning] Skipping {file_path.name}: file size "
                    f"{stat.st_size} exceeds the {RAG_MAX_FILE_BYTES}-byte limit.[/warning]"
                )
                failed_sources.add(file_key)
                continue
            if total_source_bytes + stat.st_size > RAG_MAX_CORPUS_BYTES:
                console.print(
                    f"[warning][Warning] Skipping {file_path.name}: corpus byte "
                    f"limit ({RAG_MAX_CORPUS_BYTES}) reached.[/warning]"
                )
                failed_sources.add(file_key)
                continue
            total_source_bytes += stat.st_size
            source_fingerprint = {
                **_source_fingerprint(file_path, stat),
                "chunking": chunking.signature(),
                "embedding": signature,
            }

            cached_entry = _CHUNK_CACHE.get(file_key)
            if cached_entry is not None and cached_entry[0] == source_fingerprint:
                cached_docs = list(cached_entry[1])
                if cached_docs:
                    if len(all_parsed_docs) + len(cached_docs) > RAG_MAX_DOCUMENTS:
                        console.print(
                            f"[warning][Warning] Skipping {file_path.name}: document "
                            f"chunk limit ({RAG_MAX_DOCUMENTS}) would be exceeded.[/warning]"
                        )
                        failed_sources.add(file_key)
                        continue
                    all_parsed_docs.extend(cached_docs)
                updated_state[file_key] = source_fingerprint
                if (
                    file_key not in index_state
                    or index_state.get(file_key) != source_fingerprint
                ):
                    changed_docs.extend(cached_docs)
                continue

            if ext == ".pdf":
                try:
                    from pypdf import PdfReader

                    reader = PdfReader(str(file_path))
                    if len(reader.pages) > RAG_MAX_PDF_PAGES:
                        console.print(
                            f"[warning][Warning] Skipping {file_path.name}: PDF has "
                            f"{len(reader.pages)} pages; limit is {RAG_MAX_PDF_PAGES}.[/warning]"
                        )
                        failed_sources.add(file_key)
                        continue
                    pages = [
                        (page.extract_text() or "", page_number)
                        for page_number, page in enumerate(reader.pages, 1)
                    ]
                except ImportError:
                    console.print(
                        f"[warning][Warning] pypdf not installed, skipping PDF: {file_path.name}[/warning]"
                    )
                    failed_sources.add(file_key)
                    continue
            elif ext in SQLITE_EXTENSIONS:
                pages = _read_sqlite_pages(file_path)
            elif ext in {".html", ".htm"}:
                try:
                    from bs4 import BeautifulSoup

                    soup = BeautifulSoup(
                        file_path.read_text(encoding="utf-8", errors="ignore"),
                        "html.parser",
                    )
                    for element in soup(
                        ["script", "style", "nav", "footer", "header", "aside"]
                    ):
                        element.decompose()
                    pages = [(soup.get_text(separator="\n", strip=True), None)]
                except ImportError:
                    pages = [
                        (file_path.read_text(encoding="utf-8", errors="ignore"), None)
                    ]
            elif ext in {".docx", ".xlsx", ".pptx", ".epub"}:
                try:
                    import html as _html
                    import re as _re
                    import zipfile

                    texts = []
                    with zipfile.ZipFile(file_path) as archive:
                        for member in archive.namelist():
                            if member.endswith(
                                (".xml", ".html", ".xhtml", ".htm", ".txt")
                            ):
                                try:
                                    data = archive.read(member).decode(
                                        "utf-8", errors="ignore"
                                    )
                                except OSError, KeyError, RuntimeError:
                                    continue
                                text = _re.sub(r"<[^>]+>", " ", data)
                                text = _html.unescape(text)
                                if text.strip():
                                    texts.append(text)
                    pages = [("\n".join(texts), None)] if texts else []
                except OSError, RuntimeError, ValueError, zipfile.BadZipFile:
                    pages = []
            else:
                pages = [(file_path.read_text(encoding="utf-8", errors="ignore"), None)]

            file_docs = []
            chunk_number = 0
            for content, page_number in pages:
                for chunk_text in _split_into_chunks(
                    content, chunking, ext, getattr(embeddings, "embed_documents", None)
                ):
                    metadata = {
                        "source": file_key,
                        "chunk": chunk_number,
                        "collection": collection_name,
                    }
                    if page_number is not None:
                        metadata["page"] = page_number
                    chunk_id = hashlib.sha256(
                        f"{file_key}\0{chunk_number}\0{chunk_text}".encode()
                    ).hexdigest()
                    metadata["chunk_id"] = chunk_id
                    file_docs.append(
                        Document(page_content=chunk_text, metadata=metadata)
                    )
                    chunk_number += 1

            _CHUNK_CACHE[file_key] = (source_fingerprint, list(file_docs))
            if file_docs:
                if len(all_parsed_docs) + len(file_docs) > RAG_MAX_DOCUMENTS:
                    console.print(
                        f"[warning][Warning] Skipping {file_path.name}: document "
                        f"chunk limit ({RAG_MAX_DOCUMENTS}) would be exceeded.[/warning]"
                    )
                    failed_sources.add(file_key)
                    continue
                all_parsed_docs.extend(file_docs)
                updated_state[file_key] = source_fingerprint
                if (
                    file_key not in index_state
                    or index_state[file_key] != source_fingerprint
                ):
                    changed_docs.extend(file_docs)
        except (ValueError, OSError) as e:
            failed_sources.add(str(file_path))
            console.print(
                f"[warning][Warning] Failed parsing {file_path.name}: {e}[/warning]"
            )

    state_write_started = False
    try:
        if Chroma is None or embeddings is None:
            vectorstore = None
        elif db_exists:
            vectorstore = Chroma(
                persist_directory=chroma_persist_dir, embedding_function=embeddings
            )
            changed_sources = {doc.metadata["source"] for doc in changed_docs}
            old_ids_by_source = {}
            for source in changed_sources:
                previous = vectorstore.get(where={"source": source})
                old_ids_by_source[source] = set(previous.get("ids", []))
            if changed_docs:
                console.print(
                    f"[success][Incremental Indexing][/success] Embedding {len(changed_docs)} new or modified chunk(s)..."
                )
                vectorstore.add_documents(
                    changed_docs,
                    ids=[doc.metadata["chunk_id"] for doc in changed_docs],
                )
                for source in changed_sources:
                    current_ids = {
                        doc.metadata["chunk_id"]
                        for doc in changed_docs
                        if doc.metadata["source"] == source
                    }
                    obsolete_ids = old_ids_by_source[source] - current_ids
                    if obsolete_ids:
                        vectorstore.delete(ids=list(obsolete_ids))
            else:
                console.print(
                    "[success][Incremental Indexing][/success] No document changes detected. Skipping re-embedding."
                )
            removed_sources = set(index_state) - set(updated_state) - failed_sources
            for source in removed_sources:
                vectorstore.delete(where={"source": source})
        else:
            if all_parsed_docs:
                console.print(
                    f"[success][Initial Indexing][/success] Creating vector store with {len(all_parsed_docs)} chunk(s)..."
                )
                vectorstore = Chroma.from_documents(
                    all_parsed_docs,
                    embeddings,
                    persist_directory=chroma_persist_dir,
                    ids=[doc.metadata["chunk_id"] for doc in all_parsed_docs],
                )
            else:
                vectorstore = Chroma(
                    persist_directory=chroma_persist_dir, embedding_function=embeddings
                )

        if vectorstore is not None:
            for source in failed_sources:
                if source in index_state:
                    updated_state[source] = index_state[source]
            state_write_started = True
            _write_index_state_atomic(state_file_path, updated_state)
        else:
            console.print(
                f"[success][Keyword RAG][/success] Indexed {len(all_parsed_docs)} "
                "chunk(s) for keyword search."
            )

        # Return Hybrid RAG retriever wrapper
        from .rerank import reranker_from_config
        from .retrieval import HybridRAGRetriever

        return HybridRAGRetriever(
            vectorstore,
            all_parsed_docs,
            reranker=reranker_from_config(APP_CONFIG),
        )
    except (ValueError, OSError, RuntimeError) as e:
        if (
            db_exists
            and not state_write_started
            and confirm_rebuild is not None
            and confirm_rebuild(
                f"Vector database initialization/update failed: {e}. "
                "The damaged index will be preserved before rebuilding."
            )
        ):
            try:
                backup_path = _preserve_corrupt_index(pathlib.Path(chroma_persist_dir))
                console.print(
                    f"[warning][RAG recovery][/warning] Preserved the previous "
                    f"vector index at '{backup_path}'. Rebuilding from current "
                    "knowledge-base files."
                )
                if all_parsed_docs:
                    vectorstore = Chroma.from_documents(
                        all_parsed_docs,
                        embeddings,
                        persist_directory=chroma_persist_dir,
                        ids=[doc.metadata["chunk_id"] for doc in all_parsed_docs],
                    )
                else:
                    vectorstore = Chroma(
                        persist_directory=chroma_persist_dir,
                        embedding_function=embeddings,
                    )
                _write_index_state_atomic(state_file_path, updated_state)
                from .rerank import reranker_from_config
                from .retrieval import HybridRAGRetriever

                return HybridRAGRetriever(
                    vectorstore,
                    all_parsed_docs,
                    reranker=reranker_from_config(APP_CONFIG),
                )
            except (ValueError, OSError, RuntimeError) as recovery_error:
                console.print(
                    f"[error][RAG recovery error] Rebuild failed after preserving "
                    f"the previous index: {recovery_error}[/error]"
                )
                return None
        console.print(f"[error][Error] Vector DB initialization error: {e}[/error]")
        return None
