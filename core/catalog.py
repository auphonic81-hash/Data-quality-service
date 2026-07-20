"""Persistent catalog for datasets, analyses, and audit trail.

Backed by a single SQLite database stored alongside the reports directory.
Survives server restarts. Provides:
  - Dataset registration (raw file persisted on disk)
  - Versioning per Apply Fixes run (rollback capability)
  - Analysis history (every Analyze Quality run)
  - Remediation audit log (every fix applied, when, what changed)
"""
from __future__ import annotations
import json
import uuid
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
import pandas as pd


def _compute_source_system(filename: str) -> str:
    """Canonical source identifier — strips version/run IDs."""
    import re as _re
    fname = (filename or "").lower().strip()
    if fname.startswith("api_"):
        return "rest_api_" + _re.sub(r'\.csv$', '', fname[4:])
    if fname.startswith("db_"):
        return "sqlite_" + _re.sub(r'\.csv$', '', fname[3:])
    if fname.endswith(".pdf"):
        return "pdf_" + _re.sub(r'\.pdf$', '', fname)
    base = _re.sub(r'\.(csv|xlsx|xls|json|tsv)$', '', fname)
    return "file_" + base


class DatasetCatalog:
    """Persistent catalog using SQLite as the source of truth."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        filename TEXT NOT NULL,
        rows INTEGER NOT NULL,
        columns INTEGER NOT NULL,
        loaded_at TEXT NOT NULL,
        current_version INTEGER NOT NULL DEFAULT 1,
        raw_csv_path TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dataset_versions (
        version_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        csv_path TEXT NOT NULL,
        change_summary TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (dataset_id, version_number),
        FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
    );

    CREATE TABLE IF NOT EXISTS analyses (
        analysis_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        version_number INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        html_report_path TEXT,
        run_at TEXT NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
    );

    CREATE TABLE IF NOT EXISTS remediations (
        remediation_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        from_version INTEGER NOT NULL,
        to_version INTEGER NOT NULL,
        change_log_json TEXT NOT NULL,
        total_changes INTEGER NOT NULL,
        total_failures INTEGER NOT NULL,
        applied_at TEXT NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
    );

    CREATE INDEX IF NOT EXISTS idx_versions_dataset
        ON dataset_versions(dataset_id, version_number);
    CREATE INDEX IF NOT EXISTS idx_analyses_dataset
        ON analyses(dataset_id, run_at);
    CREATE INDEX IF NOT EXISTS idx_remediations_dataset
        ON remediations(dataset_id, applied_at);

    CREATE TABLE IF NOT EXISTS archived_rows (
        archive_id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL,
        original_row_index INTEGER NOT NULL,
        row_data_json TEXT NOT NULL,
        archived_at TEXT NOT NULL,
        reason TEXT NOT NULL,
        related_dataset_id TEXT,
        related_row_index INTEGER,
        id_column TEXT,
        id_value TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
    );

    CREATE INDEX IF NOT EXISTS idx_archive_dataset
        ON archived_rows(dataset_id, archived_at);
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(self.SCHEMA)

    # ─── Datasets ─────────────────────────────────────────────────────────

    def register_dataset(
        self, source: str, filename: str, rows: int, columns: int, raw_csv_path: Path
    , source_system: str | None = None) -> str:
        """Register a newly-uploaded dataset. Returns the dataset_id."""
        now = _utc_now()
        with self._connect() as conn:
            # Idempotency check: same (source, source_system) means same file
            # being re-ingested. Return the existing dataset_id and update
            # its bookkeeping fields instead of creating a duplicate row.
            existing = None
            if source_system:
                existing = conn.execute(
                    "SELECT dataset_id FROM datasets "
                    "WHERE source = ? AND source_system = ? "
                    "LIMIT 1",
                    (source, source_system),
                ).fetchone()
            if existing:
                dataset_id = existing[0]
                # Bump current_version and register a new dataset_versions row so
                # get_current_csv_path returns the FRESH csv, not the stale v1.
                cur_ver = conn.execute(
                    "SELECT current_version FROM datasets WHERE dataset_id = ?",
                    (dataset_id,),
                ).fetchone()
                new_version = (cur_ver[0] if cur_ver else 1) + 1
                conn.execute(
                    "UPDATE datasets SET rows = ?, columns = ?, loaded_at = ?, "
                    "raw_csv_path = ?, current_version = ? WHERE dataset_id = ?",
                    (rows, columns, now, str(raw_csv_path), new_version, dataset_id),
                )
                conn.execute(
                    "INSERT INTO dataset_versions (version_id, dataset_id, version_number, "
                    "csv_path, change_summary, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (uuid.uuid4().hex[:12], dataset_id, new_version, str(raw_csv_path),
                     f"Re-ingest v{new_version}", now),
                )
                print(f"[register_dataset] re-ingest of {filename!r} -> reusing {dataset_id} (new version v{new_version})")
                return dataset_id
            dataset_id = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO datasets (dataset_id, source, filename, rows, columns, "
                "loaded_at, current_version, raw_csv_path, source_system) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (dataset_id, source, filename, rows, columns, now, str(raw_csv_path), source_system),
            )
            # Version 1 is the raw upload itself
            conn.execute(
                "INSERT INTO dataset_versions (version_id, dataset_id, version_number, "
                "csv_path, change_summary, created_at) VALUES (?, ?, 1, ?, ?, ?)",
                (
                    uuid.uuid4().hex[:12],
                    dataset_id,
                    str(raw_csv_path),
                    "Initial upload",
                    now,
                ),
            )
        return dataset_id

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM datasets ORDER BY loaded_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset and all related rows. Children deleted before parent."""
        with self._connect() as conn:
            # Delete children FIRST so FK constraints don't block the parent delete
            conn.execute(
                "DELETE FROM dataset_versions WHERE dataset_id = ?", (dataset_id,)
            )
            conn.execute(
                "DELETE FROM analyses WHERE dataset_id = ?", (dataset_id,)
            )
            conn.execute(
                "DELETE FROM remediations WHERE dataset_id = ?", (dataset_id,)
            )
            cursor = conn.execute(
                "DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,)
            )
            return cursor.rowcount > 0

    # ─── Versions (for rollback) ──────────────────────────────────────────

    def add_version(
        self, dataset_id: str, csv_path: Path, change_summary: str
    ) -> int:
        """Add a new version of a dataset and return its version_number."""
        with self._connect() as conn:
            current = conn.execute(
                "SELECT current_version FROM datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if not current:
                raise KeyError(f"Dataset {dataset_id} not found")
            new_version = current[0] + 1
            conn.execute(
                "INSERT INTO dataset_versions (version_id, dataset_id, version_number, "
                "csv_path, change_summary, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    uuid.uuid4().hex[:12],
                    dataset_id,
                    new_version,
                    str(csv_path),
                    change_summary,
                    _utc_now(),
                ),
            )
            conn.execute(
                "UPDATE datasets SET current_version = ? WHERE dataset_id = ?",
                (new_version, dataset_id),
            )
        return new_version

    def get_current_csv_path(self, dataset_id: str) -> Path | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT dv.csv_path FROM dataset_versions dv "
                "JOIN datasets d ON d.dataset_id = dv.dataset_id "
                "WHERE d.dataset_id = ? AND dv.version_number = d.current_version",
                (dataset_id,),
            ).fetchone()
            return Path(row[0]) if row else None

    def list_versions(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM dataset_versions WHERE dataset_id = ? "
                "ORDER BY version_number DESC",
                (dataset_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def rollback_to_version(self, dataset_id: str, version_number: int) -> bool:
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM dataset_versions "
                "WHERE dataset_id = ? AND version_number = ?",
                (dataset_id, version_number),
            ).fetchone()
            if not exists:
                return False
            conn.execute(
                "UPDATE datasets SET current_version = ? WHERE dataset_id = ?",
                (version_number, dataset_id),
            )
        return True

    # ─── Analyses ─────────────────────────────────────────────────────────

    def record_analysis(
        self, dataset_id: str, version_number: int,
        payload: dict[str, Any], html_report_path: str | None,
    ) -> str:
        analysis_id = uuid.uuid4().hex[:12]
        # Strip non-serializable bits from payload before storage
        safe_payload = _strip_unserializable(payload)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO analyses (analysis_id, dataset_id, version_number, "
                "payload_json, html_report_path, run_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    analysis_id, dataset_id, version_number,
                    json.dumps(safe_payload, ensure_ascii=False, default=str),
                    html_report_path, _utc_now(),
                ),
            )
        return analysis_id

    def list_analyses(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT analysis_id, version_number, html_report_path, run_at "
                "FROM analyses WHERE dataset_id = ? ORDER BY run_at DESC",
                (dataset_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── Remediations (audit trail) ───────────────────────────────────────

    def record_remediation(
        self, dataset_id: str, from_version: int, to_version: int,
        change_log: dict[str, Any], total_changes: int, total_failures: int,
    ) -> str:
        remediation_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO remediations (remediation_id, dataset_id, from_version, "
                "to_version, change_log_json, total_changes, total_failures, applied_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    remediation_id, dataset_id, from_version, to_version,
                    json.dumps(change_log, ensure_ascii=False, default=str),
                    total_changes, total_failures, _utc_now(),
                ),
            )
        return remediation_id

    def list_remediations(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM remediations WHERE dataset_id = ? "
                "ORDER BY applied_at DESC",
                (dataset_id,),
            ).fetchall()
            return [dict(r) for r in rows]




    def update_raw_csv_path(self, dataset_id: str, new_path: Path) -> None:
        """Update where a dataset\'s CSV lives on disk (catalog + v1 version row)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE datasets SET raw_csv_path=? WHERE dataset_id=?",
                (str(new_path), dataset_id),
            )
            conn.execute(
                "UPDATE dataset_versions SET csv_path=? "
                "WHERE dataset_id=? AND version_number=1",
                (str(new_path), dataset_id),
            )

    def latest_analysis(self, dataset_id: str) -> dict[str, Any] | None:
        """Return the most recent analysis payload for a dataset, or None."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT payload_json FROM analyses WHERE dataset_id = ? "
                "ORDER BY analyzed_at DESC LIMIT 1",
                (dataset_id,),
            ).fetchone()
            if not row:
                return None
            try:
                return json.loads(row["payload_json"])
            except Exception:
                return None

    # ─── Archived rows ────────────────────────────────────────────────────

    def archive_rows(
        self, dataset_id: str, rows: list[dict[str, Any]],
        reason: str = "duplicate",
    ) -> list[str]:
        """Save rows to the archive. Returns the archive_ids created.

        Each entry in `rows` should be a dict with:
          - original_row_index (int)
          - row_data (dict)
          - related_dataset_id (optional)
          - related_row_index (optional)
          - id_column (optional)
          - id_value (optional)
        """
        archive_ids: list[str] = []
        now = _utc_now()
        with self._connect() as conn:
            for r in rows:
                archive_id = uuid.uuid4().hex[:12]
                conn.execute(
                    "INSERT INTO archived_rows ("
                    "archive_id, dataset_id, original_row_index, row_data_json, "
                    "archived_at, reason, related_dataset_id, related_row_index, "
                    "id_column, id_value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        archive_id, dataset_id,
                        int(r["original_row_index"]),
                        json.dumps(r["row_data"], ensure_ascii=False, default=str),
                        now, reason,
                        r.get("related_dataset_id"),
                        r.get("related_row_index"),
                        r.get("id_column"),
                        r.get("id_value"),
                    ),
                )
                archive_ids.append(archive_id)
        return archive_ids

    def list_archived_rows(self, dataset_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM archived_rows WHERE dataset_id = ? "
                "ORDER BY archived_at DESC", (dataset_id,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["row_data"] = json.loads(d["row_data_json"])
                except Exception:
                    d["row_data"] = {}
                del d["row_data_json"]
                out.append(d)
            return out

    # ─── Connection helper ───────────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _strip_unserializable(obj: Any) -> Any:
    """Recursively drop non-JSON-serializable values (DataFrames, Reports, etc.)."""
    if isinstance(obj, dict):
        return {k: _strip_unserializable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_strip_unserializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)  # fallback: stringify
