"""Multi-format data ingestion layer.

Supports CSV, TSV, Excel, JSON, Parquet, SQLite, and HTTP sources.
"""
from __future__ import annotations

import json
import sqlite3
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


class DataIngestion:
    """Loads data from any supported source into a pandas DataFrame."""

    SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}

    @staticmethod
    def from_file(file_path: str | Path) -> pd.DataFrame:
        """Load data from a file, auto-detecting format from extension."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in DataIngestion.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported: {DataIngestion.SUPPORTED_EXTENSIONS}"
            )

        try:
            if ext == ".csv":
                return DataIngestion._read_csv(path)
            if ext == ".tsv":
                return pd.read_csv(path, sep="\t")
            if ext in {".xlsx", ".xls"}:
                return pd.read_excel(path)
            if ext == ".json":
                return DataIngestion._read_json(path)
            if ext == ".parquet":
                return pd.read_parquet(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {path.name}: {exc}") from exc

        raise ValueError(f"Unhandled format: {ext}")

    @staticmethod
    def from_sqlite(db_path: str | Path, table: str) -> pd.DataFrame:
        """Load data from a SQLite table."""
        path = Path(db_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")

        with sqlite3.connect(path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if not cursor.fetchone():
                raise ValueError(f"Table '{table}' not found in {path.name}")
            return pd.read_sql_query(f'SELECT * FROM "{table}"', conn)

    @staticmethod
    def from_url(url: str, timeout: int = 30) -> pd.DataFrame:
        """Load data from a URL (auto-detects JSON or CSV)."""
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()

        if "json" in content_type or url.endswith(".json"):
            data = response.json()
            return pd.json_normalize(data if isinstance(data, list) else [data])
        return pd.read_csv(StringIO(response.text))

    @staticmethod
    def list_sqlite_tables(db_path: str | Path) -> list[str]:
        """List all tables in a SQLite database."""
        path = Path(db_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")
        with sqlite3.connect(path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """Read CSV with encoding fallback (UTF-8 → Latin-1 → Arabic)."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1256"]
        last_error: Exception | None = None
        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc
                continue
        raise ValueError(
            f"Could not decode {path.name} with any supported encoding. Last error: {last_error}"
        )

    @staticmethod
    def _read_json(path: Path) -> pd.DataFrame:
        """Read JSON file, handling list and dict structures."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict):
            return pd.json_normalize([data])
        raise ValueError(f"Unsupported JSON structure in {path.name}")