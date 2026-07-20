"""File hashing module for duplicate detection.

Two hashes per file:

1. file_hash (SHA-256 of raw bytes):
   Catches byte-for-byte duplicates regardless of filename.
   Computed for ALL files.

2. content_hash (SHA-256 of parsed/normalized content):
   Catches semantic duplicates — same data with different column order,
   different row order, or different file format. Only computed for
   structured formats (CSV, Excel, TSV, JSON, Parquet).

Stored in file_hashes table. Path is unique (one row per path); hash
columns are indexed for fast duplicate lookup.

Enterprise behavior: first-seen wins. We track when a hash was first
observed and which path is its canonical location. New copies become
candidates for soft-delete.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# Formats we can compute a meaningful content hash for
STRUCTURED_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}

# Chunk size for streaming file hash (16 KB)
CHUNK_SIZE = 16 * 1024


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------

def compute_file_hash(path: str | Path) -> str:
    """SHA-256 of file bytes. Streamed in chunks so large files don't blow memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_content_hash(path: str | Path) -> str | None:
    """SHA-256 of normalized structured content.

    Normalization:
      - Read with pandas
      - Sort columns alphabetically
      - Convert every cell to string, strip whitespace, lowercase
      - Sort rows by the resulting all-column tuple
      - Hash the resulting CSV bytes

    Returns None for unsupported formats or unreadable files.
    Same data in different column/row order produces the same hash.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in STRUCTURED_EXTS:
        return None
    try:
        if ext in (".csv", ".tsv"):
            sep = "," if ext == ".csv" else "\t"
            df = pd.read_csv(p, sep=sep, dtype=str, keep_default_na=False, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(p, dtype=str)
            df = df.fillna("")
        elif ext == ".json":
            df = pd.read_json(p, dtype=str)
            df = df.fillna("")
        elif ext == ".parquet":
            df = pd.read_parquet(p)
            df = df.astype(str).fillna("")
        else:
            return None
    except Exception:
        return None

    if df.empty:
        return None

    # Normalize: sort columns, normalize cells, sort rows
    df = df[sorted(df.columns)]
    df = df.astype(str).apply(lambda col: col.str.strip().str.lower())
    df = df.sort_values(by=list(df.columns)).reset_index(drop=True)

    h = hashlib.sha256()
    h.update(df.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def ensure_table(catalog_db_path: str | Path) -> None:
    """Create file_hashes table + indexes if missing."""
    with sqlite3.connect(str(catalog_db_path), timeout=30) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path       TEXT NOT NULL UNIQUE,
                file_hash       TEXT NOT NULL,
                content_hash    TEXT,
                file_size       INTEGER NOT NULL,
                file_name       TEXT,
                mtime           REAL,
                first_seen_at   TEXT NOT NULL,
                last_checked_at TEXT NOT NULL,
                soft_deleted    INTEGER DEFAULT 0,
                source_label    TEXT
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_fh_file_hash ON file_hashes(file_hash)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_fh_content_hash ON file_hashes(content_hash)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_fh_soft_deleted ON file_hashes(soft_deleted)")


def upsert_hash(
    catalog_db_path: str | Path,
    file_path: str | Path,
    source_label: str = "unknown",
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Compute hash for a file and store/refresh it. Returns hash info.

    If a row already exists for this path AND file mtime hasn't changed AND
    force_recompute is False, we skip hash computation (the file hasn't been
    modified, so its hash hasn't either).

    Returns:
      {
        "file_path": str,
        "file_hash": str,
        "content_hash": str | None,
        "file_size": int,
        "is_duplicate": bool,            # same file_hash exists at a different path
        "is_content_duplicate": bool,    # same content_hash exists at a different path
        "duplicate_of": [str, ...],      # other paths sharing this file_hash
        "content_duplicate_of": [str, ...],
        "first_seen_at": str,
      }
    """
    ensure_table(catalog_db_path)
    p = Path(file_path).resolve()
    if not p.exists() or not p.is_file():
        return {"error": f"file not found: {p}"}

    stat = p.stat()
    mtime = stat.st_mtime
    size = stat.st_size
    now = datetime.utcnow().isoformat()

    with sqlite3.connect(str(catalog_db_path), timeout=30) as con:
        con.row_factory = sqlite3.Row
        existing = con.execute(
            "SELECT * FROM file_hashes WHERE file_path = ?", (str(p),)
        ).fetchone()

        if existing and not force_recompute and existing["mtime"] == mtime:
            file_hash = existing["file_hash"]
            content_hash = existing["content_hash"]
            first_seen = existing["first_seen_at"]
            # Refresh last_checked_at only
            con.execute(
                "UPDATE file_hashes SET last_checked_at = ? WHERE file_path = ?",
                (now, str(p)),
            )
        else:
            file_hash = compute_file_hash(p)
            content_hash = compute_content_hash(p)
            if existing:
                con.execute(
                    "UPDATE file_hashes SET file_hash = ?, content_hash = ?, "
                    "file_size = ?, mtime = ?, last_checked_at = ? "
                    "WHERE file_path = ?",
                    (file_hash, content_hash, size, mtime, now, str(p)),
                )
                first_seen = existing["first_seen_at"]
            else:
                con.execute(
                    "INSERT INTO file_hashes "
                    "(file_path, file_hash, content_hash, file_size, file_name, "
                    " mtime, first_seen_at, last_checked_at, source_label) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(p), file_hash, content_hash, size, p.name,
                     mtime, now, now, source_label),
                )
                first_seen = now

        # Look up other paths sharing this hash (active only — exclude soft-deleted)
        dups = con.execute(
            "SELECT file_path FROM file_hashes "
            "WHERE file_hash = ? AND file_path != ? AND soft_deleted = 0 "
            "ORDER BY first_seen_at",
            (file_hash, str(p)),
        ).fetchall()
        content_dups = []
        if content_hash:
            content_dups = con.execute(
                "SELECT file_path FROM file_hashes "
                "WHERE content_hash = ? AND file_path != ? AND soft_deleted = 0 "
                "ORDER BY first_seen_at",
                (content_hash, str(p)),
            ).fetchall()

    return {
        "file_path":             str(p),
        "file_hash":             file_hash,
        "content_hash":          content_hash,
        "file_size":             size,
        "is_duplicate":          len(dups) > 0,
        "is_content_duplicate":  len(content_dups) > 0,
        "duplicate_of":          [r["file_path"] for r in dups],
        "content_duplicate_of":  [r["file_path"] for r in content_dups],
        "first_seen_at":         first_seen,
    }


def get_hash_info(catalog_db_path: str | Path, file_path: str | Path) -> dict[str, Any] | None:
    """Look up hash info for a file path without computing anything."""
    ensure_table(catalog_db_path)
    with sqlite3.connect(str(catalog_db_path), timeout=30) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM file_hashes WHERE file_path = ?", (str(Path(file_path).resolve()),)
        ).fetchone()
    return dict(row) if row else None


def find_duplicate_groups(catalog_db_path: str | Path, by: str = "file_hash") -> list[dict[str, Any]]:
    """Return all duplicate groups in the database.

    by: 'file_hash' (default) or 'content_hash'

    Each group is: {hash, files: [{path, size, first_seen_at}, ...], canonical}
    canonical = the OLDEST file in the group (first-seen wins).
    """
    if by not in ("file_hash", "content_hash"):
        raise ValueError(f"invalid 'by' parameter: {by}")
    ensure_table(catalog_db_path)
    with sqlite3.connect(str(catalog_db_path), timeout=30) as con:
        con.row_factory = sqlite3.Row
        # Find hashes that appear 2+ times among non-deleted files
        hashes = con.execute(
            f"SELECT {by} as h, COUNT(*) as n FROM file_hashes "
            f"WHERE soft_deleted = 0 AND {by} IS NOT NULL "
            f"GROUP BY {by} HAVING n > 1"
        ).fetchall()
        groups = []
        for hrow in hashes:
            files = con.execute(
                f"SELECT file_path, file_size, file_name, first_seen_at "
                f"FROM file_hashes WHERE {by} = ? AND soft_deleted = 0 "
                f"ORDER BY first_seen_at",
                (hrow["h"],),
            ).fetchall()
            file_list = [dict(f) for f in files]
            groups.append({
                "hash":      hrow["h"],
                "hash_type": by,
                "count":     len(file_list),
                "canonical": file_list[0]["file_path"],  # oldest wins
                "duplicates": [f["file_path"] for f in file_list[1:]],
                "files":     file_list,
            })
    return groups


def scan_directory(
    catalog_db_path: str | Path,
    directory: str | Path,
    source_label: str = "scan",
    recursive: bool = False,
) -> dict[str, Any]:
    """Hash every file in a directory and record results.

    Returns: {scanned: N, new: N, refreshed: N, duplicates_found: N, groups: [...]}
    """
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        return {"error": f"directory not found: {d}"}

    files = list(d.rglob("*") if recursive else d.iterdir())
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    scanned = 0
    new_count = 0
    refreshed = 0
    for f in files:
        before = get_hash_info(catalog_db_path, f)
        result = upsert_hash(catalog_db_path, f, source_label=source_label)
        if "error" in result:
            continue
        scanned += 1
        if before is None:
            new_count += 1
        else:
            refreshed += 1

    # Now compute duplicate groups across the whole database
    file_groups = find_duplicate_groups(catalog_db_path, by="file_hash")
    content_groups = find_duplicate_groups(catalog_db_path, by="content_hash")

    return {
        "scanned":           scanned,
        "new":               new_count,
        "refreshed":         refreshed,
        "file_hash_groups":  file_groups,
        "content_hash_groups": content_groups,
        "duplicates_found":  sum(g["count"] - 1 for g in file_groups),
    }