"""Pre-ingestion duplicate detection via content hash.

When the same data appears in multiple files (different filenames, different
formats, different column orders), we don't want to ingest it twice. The
content_hash (SHA-256 of normalized rows) catches semantic duplicates that
file_hash misses.

Behavior:
  1. Compute content_hash for the incoming file
  2. Look up that hash in file_hashes table
  3. If found at a DIFFERENT, non-deleted path → it's a duplicate
  4. Soft-delete the new file to ~/data-quality-service/duplicates/
  5. Log to file_quarantine with reason_category='duplicate_skipped'
  6. Return a result dict the caller can include in its response

PDFs and other unstructured files have content_hash=None — for those the
check returns None and the caller proceeds with normal PK-based row dedup.
"""
from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


DUPLICATES_DIR = Path.home() / "data-quality-service" / "duplicates"


def ensure_duplicates_setup(catalog_db_path: str | Path) -> None:
    """Create the duplicates/ directory and (reuse) file_quarantine table."""
    DUPLICATES_DIR.mkdir(parents=True, exist_ok=True)
    # file_quarantine is created by file_disposition.ensure_aside_setup,
    # but make sure it exists here too in case dedup runs before any aside.
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


def move_to_duplicates(
    source_path: str | Path,
    canonical_path: str,
    content_hash: str,
    catalog_db_path: str | Path,
) -> dict[str, Any]:
    """Soft-delete a duplicate source file: move to duplicates/ + sidecar + log.

    Returns: {disposition, duplicates_path, sidecar_path, moved_at}
    """
    ensure_duplicates_setup(catalog_db_path)
    src = Path(source_path)
    if not src.exists():
        return {"disposition": "kept", "reason": "file already gone"}

    now = datetime.utcnow().isoformat()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dup_name = f"{src.stem}_{timestamp}{src.suffix}"
    dup_path = DUPLICATES_DIR / dup_name

    # Avoid collision
    counter = 1
    while dup_path.exists():
        dup_path = DUPLICATES_DIR / f"{src.stem}_{timestamp}_{counter}{src.suffix}"
        counter += 1

    shutil.move(str(src), str(dup_path))

    # Sidecar explaining the dedup decision
    sidecar = dup_path.with_suffix(dup_path.suffix + ".duplicate_of.txt")
    sidecar.write_text(
        f"This file was soft-deleted because its content matches "
        f"an existing file in the system.\n\n"
        f"Original path:    {src}\n"
        f"Moved to:         {dup_path}\n"
        f"Moved at (UTC):   {now}\n"
        f"Duplicate of:     {canonical_path}\n"
        f"Match basis:      content_hash (SHA-256 of normalized rows)\n"
        f"content_hash:     {content_hash}\n\n"
        f"This means the same data — possibly in a different column or row "
        f"order, or even a different file format — was already ingested.\n"
        f"To restore: move the file back to the original path.\n",
        encoding="utf-8",
    )

    # Log to file_quarantine with the duplicate category
    try:
        with sqlite3.connect(str(catalog_db_path), timeout=30) as con:
            con.execute(
                "INSERT INTO file_quarantine "
                "(original_path, aside_path, reason_category, reason_details, moved_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(src),
                    str(dup_path),
                    "duplicate_skipped",
                    f"Content matches canonical file: {canonical_path} (content_hash: {content_hash[:12]})",
                    now,
                ),
            )
            # Also mark this file as soft_deleted in file_hashes so it can't win
            # the canonical race for its own content_hash on future ingestions.
            con.execute(
                "UPDATE file_hashes SET soft_deleted = 1 WHERE file_path = ?",
                (str(src),),
            )
    except Exception as e:
        print(f"[dedup] WARN: could not log to file_quarantine: {e}")

    print(f"[dedup] SOFT-DELETED {src.name} -> duplicates/ (matches {Path(canonical_path).name})")
    return {
        "disposition":    "soft_deleted_to_duplicates",
        "duplicates_path": str(dup_path),
        "sidecar_path":   str(sidecar),
        "moved_at":       now,
    }


def check_pre_ingestion_duplicate(
    file_path: str | Path,
    source_type: str,
    catalog_db_path: str | Path,
) -> dict[str, Any] | None:
    """Run before ingesting a file. Decide whether to proceed or skip.

    Returns None → proceed with normal ingestion (no semantic duplicate found,
    OR file is unstructured so content_hash isn't computable).

    Returns dict → skip ingestion. The dict has:
      - status:        'duplicate_skipped'
      - reason:        explanation string
      - canonical:     path of the original file
      - content_hash:  short hash for display
      - disposition:   result of move_to_duplicates() ('soft_deleted_to_duplicates')

    Skips DB/REST sources (no file on disk).
    Skips files inside the project's uploads/ working dir (those are working copies).
    """
    # Skip when there's no file to check
    if source_type in ("database", "restapi"):
        return None
    src = Path(file_path)
    if not src.exists():
        return None
    # Don't touch internal working copies
    try:
        uploads_dir = (Path.home() / "data-quality-service" / "uploads").resolve()
        if src.resolve().is_relative_to(uploads_dir):
            return None
    except Exception:
        pass

    try:
        from core.file_hash import upsert_hash
    except Exception as e:
        print(f"[dedup] file_hash import failed: {e}")
        return None

    info = upsert_hash(catalog_db_path, src, source_label=f"pre_ingest_{source_type}")
    if "error" in info:
        return None

    content_hash = info.get("content_hash")
    if not content_hash:
        # Unstructured file (PDF, image, etc) — let normal ingestion handle dedup
        return None

    if not info.get("is_content_duplicate"):
        return None

    # We have a semantic duplicate. Find the canonical (oldest) path.
    canonical = None
    content_dups = info.get("content_duplicate_of", [])
    if content_dups:
        # The list is ordered by first_seen_at in upsert_hash, so first item = oldest
        canonical = content_dups[0]
    if not canonical:
        return None

    # Soft-delete the new file
    dispo = move_to_duplicates(src, canonical, content_hash, catalog_db_path)

    return {
        "status":             "duplicate_skipped",
        "reason":             "content matches existing file in system",
        "canonical":          canonical,
        "content_hash":       content_hash,
        "content_hash_short": content_hash[:12],
        "disposition":        dispo.get("disposition", "kept"),
        "duplicates_path":    dispo.get("duplicates_path"),
    }