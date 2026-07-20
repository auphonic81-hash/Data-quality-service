"""File disposition logic for the ingestion pipeline.

Decides per file what to do AFTER ingestion attempts:

- DELETED: source successfully represented in master (or confirmed
  duplicate). Source file removed from disk so it's not re-ingested.

- ASIDED: file couldn't be processed (poem, image, schema-incompatible
  document). Moved to aside/ with a .reason.txt sidecar AND a database
  log entry in file_quarantine.

- KEPT: file untouched (only for unexpected errors).

Pure stdlib. No new dependencies. Aside directory: ~/data-quality-service/aside/
"""
from __future__ import annotations
import shutil
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path


ASIDE_DIR   = Path.home() / "data-quality-service" / "aside"
ARCHIVE_DIR = Path.home() / "data-quality-service" / "archive"


class FileDisposition(str, Enum):
    DELETED  = "deleted"
    ARCHIVED = "archived"
    ASIDED  = "asided"
    KEPT    = "kept"


def ensure_aside_setup(catalog_db_path: str | Path) -> None:
    """Create aside/ directory and file_quarantine table if missing."""
    ASIDE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(catalog_db_path), timeout=30) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS file_quarantine (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                original_path   TEXT NOT NULL,
                aside_path      TEXT NOT NULL,
                reason_category TEXT NOT NULL,
                reason_details  TEXT,
                moved_at        TEXT NOT NULL
            )
        """)
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_quarantine_moved_at "
            "ON file_quarantine(moved_at)"
        )


def move_to_aside(
    source_path: str | Path,
    catalog_db_path: str | Path,
    reason_category: str,
    reason_details: str = "",
) -> FileDisposition:
    """Move file to aside/ with sidecar reason + database log.

    reason_category is short (e.g. 'unrecognized_entity', 'all_rows_rejected').
    reason_details is the human-readable explanation.
    """
    ensure_aside_setup(catalog_db_path)
    source = Path(source_path)
    if not source.exists():
        # File already gone — log it but no move needed
        _log_only(catalog_db_path, str(source), "", reason_category,
                  reason_details + " (file already missing)")
        return FileDisposition.KEPT

    now = datetime.utcnow().isoformat()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    aside_name = f"{source.stem}_{timestamp}{source.suffix}"
    aside_path = ASIDE_DIR / aside_name

    # Avoid collision if same file asided twice in same second
    counter = 1
    while aside_path.exists():
        aside_path = ASIDE_DIR / f"{source.stem}_{timestamp}_{counter}{source.suffix}"
        counter += 1

    shutil.move(str(source), str(aside_path))

    # Sidecar reason file: human-readable, sits next to the asided file
    sidecar_path = aside_path.with_suffix(aside_path.suffix + ".reason.txt")
    sidecar_path.write_text(
        f"Original path: {source}\n"
        f"Aside location: {aside_path}\n"
        f"Moved at (UTC): {now}\n"
        f"Reason category: {reason_category}\n"
        f"Reason details:\n{reason_details}\n",
        encoding="utf-8",
    )

    _log_only(catalog_db_path, str(source), str(aside_path),
              reason_category, reason_details)
    print(f"[disposition] ASIDED {source.name} → {aside_path.name} ({reason_category})")
    return FileDisposition.ASIDED


def _log_only(catalog_db_path, original_path, aside_path, reason_category, reason_details):
    """Insert a row into file_quarantine."""
    now = datetime.utcnow().isoformat()
    try:
        with sqlite3.connect(str(catalog_db_path), timeout=30) as con:
            con.execute(
                "INSERT INTO file_quarantine "
                "(original_path, aside_path, reason_category, reason_details, moved_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (original_path, aside_path, reason_category, reason_details, now),
            )
    except Exception as e:
        print(f"[disposition] WARN: could not log to file_quarantine: {e}")


def archive_source(source_path: str | Path) -> FileDisposition:
    """Move a successfully-ingested source file to the archive directory.

    Archive layout: ~/data-quality-service/archive/YYYY-MM-DD/<original_name>
    Sidecar: <archived_name>.archived_at.txt records original path + timestamp.
    Preserves audit trail; source folder empties.
    """
    source = Path(source_path)
    if not source.exists():
        return FileDisposition.KEPT
    day_dir = ARCHIVE_DIR / datetime.utcnow().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.utcnow().isoformat()
    dest = day_dir / source.name
    counter = 1
    while dest.exists():
        dest = day_dir / f"{source.stem}_{counter}{source.suffix}"
        counter += 1
    try:
        shutil.move(str(source), str(dest))
        (dest.with_suffix(dest.suffix + ".archived_at.txt")).write_text(
            f"Original path: {source}\nArchived to:   {dest}\n"
            f"Archived at (UTC): {now}\nReason: ingested successfully into master\n",
            encoding="utf-8",
        )
        print(f"[disposition] ARCHIVED {source.name} -> {dest}")
    except Exception as e:
        print(f"[disposition] WARN: could not archive {source}: {e}")
        return FileDisposition.KEPT
    return FileDisposition.ARCHIVED


def decide_disposition(
    source_path: str | Path,
    catalog_db_path: str | Path,
    ingest_result: dict | None,
    target_entity: str | None,
    parse_error: str | None = None,
) -> FileDisposition:
    """Apply the disposition rules after an ingestion attempt.

    Decision tree:
      1. parse_error is set         → ASIDE (file couldn't be opened/parsed)
      2. target_entity is None      → ASIDE (classifier couldn't identify)
      3. Any rows landed/merged/enriched → DELETE (file represented in master)
      4. All rows rejected by schema → ASIDE (file shape doesn't match)
      5. No rows produced at all    → ASIDE (empty or unparseable structure)
    """
    if parse_error:
        return move_to_aside(
            source_path, catalog_db_path,
            reason_category="parse_error",
            reason_details=f"Could not parse file: {parse_error}",
        )

    if target_entity is None:
        return move_to_aside(
            source_path, catalog_db_path,
            reason_category="unrecognized_entity",
            reason_details=(
                "The entity classifier could not identify a known entity "
                "type for this file. It may be unrelated to MDM domains "
                "(e.g., a poem, image, marketing PDF, or arbitrary document)."
            ),
        )

    ing = ingest_result or {}
    inserted = int(ing.get("rows_inserted", 0) or 0)
    merged   = int(ing.get("rows_duplicates_found", 0) or ing.get("rows_merged", 0) or 0)
    enriched = int(ing.get("fields_enriched_total", 0) or 0)
    rejected = int(ing.get("rows_rejected", 0) or 0) + int(ing.get("rows_rejected_by_schema", 0) or 0)

    if inserted > 0 or merged > 0 or enriched > 0:
        return archive_source(source_path)

    if rejected > 0:
        return move_to_aside(
            source_path, catalog_db_path,
            reason_category="all_rows_rejected",
            reason_details=(
                f"Schema validation rejected all {rejected} rows. "
                f"Entity: {target_entity}. See master_rejections_v2 "
                f"for per-row reasons. The file may use a different "
                f"vocabulary or have malformed data."
            ),
        )

    return move_to_aside(
        source_path, catalog_db_path,
        reason_category="no_data_extracted",
        reason_details=(
            f"File was parsed without errors but produced no rows for "
            f"entity '{target_entity}'. May be empty, or the parser "
            f"could not find structured data."
        ),
    )


def list_aside_files(catalog_db_path: str | Path) -> list[dict]:
    """Return contents of file_quarantine for UI display."""
    ensure_aside_setup(catalog_db_path)
    with sqlite3.connect(str(catalog_db_path), timeout=30) as con:
        rows = con.execute(
            "SELECT original_path, aside_path, reason_category, "
            "reason_details, moved_at FROM file_quarantine "
            "ORDER BY moved_at DESC LIMIT 200"
        ).fetchall()
    return [
        {
            "original_path":   r[0],
            "aside_path":      r[1],
            "reason_category": r[2],
            "reason_details":  r[3],
            "moved_at":        r[4],
        }
        for r in rows
    ]