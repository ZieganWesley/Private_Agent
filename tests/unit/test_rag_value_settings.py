"""RAG file types and SQLite chunking values come from settings (K5/14.7)."""

from __future__ import annotations

import sqlite3

import config
from config import DEFAULT_CONFIG
from config.settings import Settings

DEFAULT_FILE_TYPES = {
    ".pdf",
    ".txt",
    ".md",
    ".py",
    ".json",
    ".csv",
    ".rs",
    ".js",
    ".ts",
    ".html",
    ".htm",
    ".docx",
    ".xlsx",
    ".pptx",
    ".epub",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".db3",
}


def _settings(rag: dict | None = None) -> Settings:
    return Settings.from_config(
        {"rag": rag or {}}, default_paths=DEFAULT_CONFIG["paths"]
    )


def test_rag_values_default_to_the_previous_fixed_values() -> None:
    settings = _settings()

    assert set(settings.rag_file_types) == DEFAULT_FILE_TYPES
    assert settings.rag_sqlite_rows_per_chunk == 10
    assert settings.rag_sqlite_max_cell_chars == 200


def test_rag_file_types_are_normalized_and_bounded() -> None:
    settings = _settings(
        {
            "file_types": [
                "TXT",
                ".MD",
                "csv",
                "",
                None,
                5,
                ".not a suffix",
                "." + "x" * 20,
            ]
        }
    )

    assert settings.rag_file_types == [".txt", ".md", ".csv"]


def test_empty_or_invalid_rag_file_types_fall_back_to_the_defaults() -> None:
    assert set(_settings({"file_types": []}).rag_file_types) == DEFAULT_FILE_TYPES
    assert set(_settings({"file_types": "pdf"}).rag_file_types) == DEFAULT_FILE_TYPES
    assert set(_settings({"file_types": [""]}).rag_file_types) == DEFAULT_FILE_TYPES


def test_sqlite_chunking_values_are_configurable_and_bounded() -> None:
    settings = _settings({"sqlite_rows_per_chunk": 25, "sqlite_max_cell_chars": 80})
    assert settings.rag_sqlite_rows_per_chunk == 25
    assert settings.rag_sqlite_max_cell_chars == 80

    floored = _settings({"sqlite_rows_per_chunk": 0, "sqlite_max_cell_chars": 1})
    assert floored.rag_sqlite_rows_per_chunk == 10
    assert floored.rag_sqlite_max_cell_chars == 200


def test_module_constants_mirror_settings() -> None:
    assert config.SETTINGS.rag_file_types == config.RAG_FILE_TYPES
    assert config.SETTINGS.rag_sqlite_rows_per_chunk == config.RAG_SQLITE_ROWS_PER_CHUNK
    assert config.SETTINGS.rag_sqlite_max_cell_chars == config.RAG_SQLITE_MAX_CELL_CHARS


def test_indexing_uses_the_configured_types_and_keeps_sqlite_support() -> None:
    from rag import indexing

    assert (
        frozenset(config.RAG_FILE_TYPES) | (indexing.SQLITE_EXTENSIONS)
        == indexing.SUPPORTED_EXTENSIONS
    )
    assert indexing.SQLITE_EXTENSIONS <= indexing.SUPPORTED_EXTENSIONS
    assert indexing._SQLITE_ROWS_PER_CHUNK == config.RAG_SQLITE_ROWS_PER_CHUNK
    assert indexing._SQLITE_MAX_CELL_CHARS == config.RAG_SQLITE_MAX_CELL_CHARS


def test_sqlite_cell_truncates_at_the_configured_width(monkeypatch) -> None:
    from rag import indexing

    monkeypatch.setattr(indexing, "_SQLITE_MAX_CELL_CHARS", 10)
    assert indexing._sqlite_cell("abcdefghijklmno") == "abcdefghij..."
    assert indexing._sqlite_cell("short") == "short"
    assert indexing._sqlite_cell(None) == "NULL"


def test_sqlite_pages_batch_rows_at_the_configured_size(tmp_path, monkeypatch) -> None:
    from rag import indexing

    database = tmp_path / "sample.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (name TEXT)")
        connection.executemany(
            "INSERT INTO items (name) VALUES (?)",
            [(f"row-{index}",) for index in range(25)],
        )

    monkeypatch.setattr(indexing, "_SQLITE_ROWS_PER_CHUNK", 7)
    pages = indexing._read_sqlite_pages(database)

    assert len(pages) == 4
    assert [page[0].count("row-") for page in pages] == [7, 7, 7, 4]
