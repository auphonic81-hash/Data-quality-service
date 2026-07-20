"""OpenLineage event emitter for the Unified Data Platform.

Emits standard OpenLineage events for every ingest run. Events are stored
locally in SQLite and exportable via the /api/openlineage endpoint.

Reference: https://openlineage.io/docs/spec/object-model/
"""
from __future__ import annotations
import uuid
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.run import (
        RunEvent, RunState, Run, Job, Dataset,
    )
    OPENLINEAGE_AVAILABLE = True
except ImportError:
    OPENLINEAGE_AVAILABLE = False


_PRODUCER = "https://github.com/auphonic81-hash/Data-quality-service"
_NAMESPACE = "unified-data-platform"


def _utc_iso():
    return datetime.now(timezone.utc).isoformat()


class OpenLineageEmitter:
    """Captures lineage events in the catalog for export to Marquez/DataHub."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def emit_ingest_run(
        self,
        source_dataset_id: str,
        source_name: str,
        target_master_table: str,
        rows_inserted: int,
        rows_duplicates_found: int,
        fields_enriched_total: int,
    ) -> str:
        """Record a complete ingest run as 2 events: START + COMPLETE."""
        run_id = str(uuid.uuid4())
        job_name = f"ingest_{target_master_table.lower()}"
        now = _utc_iso()

        # Build standard OpenLineage payloads
        start_event = self._build_event(
            event_type="START",
            run_id=run_id,
            job_name=job_name,
            event_time=now,
            inputs=[{"namespace": _NAMESPACE, "name": source_name}],
            outputs=[],
        )
        complete_event = self._build_event(
            event_type="COMPLETE",
            run_id=run_id,
            job_name=job_name,
            event_time=now,
            inputs=[{"namespace": _NAMESPACE, "name": source_name}],
            outputs=[{
                "namespace": _NAMESPACE,
                "name": f"master_{target_master_table.lower()}",
                "facets": {
                    "statistics": {
                        "rowsInserted":   rows_inserted,
                        "rowsMerged":     rows_duplicates_found,
                        "fieldsEnriched": fields_enriched_total,
                    }
                },
            }],
        )

        # Store both events
        with sqlite3.connect(self.db_path, timeout=30) as con:
            for ev in (start_event, complete_event):
                con.execute(
                    "INSERT INTO openlineage_events "
                    "(event_id, event_type, event_time, run_id, job_name, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(uuid.uuid4()),
                        ev["eventType"],
                        ev["eventTime"],
                        run_id,
                        job_name,
                        json.dumps(ev),
                    ),
                )
        return run_id

    @staticmethod
    def _build_event(event_type, run_id, job_name, event_time, inputs, outputs):
        return {
            "eventType":  event_type,
            "eventTime":  event_time,
            "producer":   _PRODUCER,
            "schemaURL":  "https://openlineage.io/spec/2-0-2/OpenLineage.json",
            "run":        {"runId": run_id},
            "job":        {"namespace": _NAMESPACE, "name": job_name},
            "inputs":     inputs,
            "outputs":    outputs,
        }

    def export_all(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return all stored events (latest first) for the export API."""
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT payload FROM openlineage_events ORDER BY event_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [json.loads(r["payload"]) for r in rows]

    def count(self) -> int:
        with sqlite3.connect(self.db_path, timeout=30) as con:
            return con.execute("SELECT COUNT(*) FROM openlineage_events").fetchone()[0]
