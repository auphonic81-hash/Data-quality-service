"""Data Quality Service — Flask REST API + Dashboard.

Endpoints:
  GET  /                          → Dashboard UI
  POST /api/upload                → Upload a file, returns dataset_id
  POST /api/load-sqlite           → Load a SQLite table, returns dataset_id
  GET  /api/datasets              → List cached datasets
  GET  /api/datasets/<id>         → Get dataset metadata
  POST /api/datasets/<id>/analyze → Run full analysis pipeline
  POST /api/datasets/<id>/remediate → Apply automated fixes
  GET  /api/datasets/<id>/export?format=csv → Export cleaned data
  GET  /api/datasets/<id>/report  → Serve the HTML profiling report
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

import config
from core import DataQualityService


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.json.encoder = NumpyJSONEncoder
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

service = DataQualityService(
    reports_dir=config.REPORTS_DIR,
    ollama_url=config.OLLAMA_URL,
    ollama_model=config.OLLAMA_MODEL,
)


# ─── Dashboard ─────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ─── Data loading ──────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in config.SUPPORTED_FORMATS:
        return jsonify({
            "error": f"Unsupported format '{ext}'",
            "supported": list(config.SUPPORTED_FORMATS),
        }), 400

    save_path = config.UPLOADS_DIR / filename
    file.save(save_path)

    try:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            try:
                pdf_result = service.load_from_pdf(save_path)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            # Continue into the shared auto-pipeline below using the PDF\'s dataset_id.
            dataset_id = pdf_result["dataset_id"]
            import pandas as _pd
            df = _pd.read_csv(service.catalog.get_dataset(dataset_id)["raw_csv_path"])
        else:
            dataset_id, df = service.load_from_file(save_path)
        if False:  # placeholder so following block stays unchanged
            pass
        # NOTE: code below this point was: dataset_id, df = service.load_from_file(save_path)

        # ── Auto-pipeline: profile → standardize → ingest into master ──
        # Falls through gracefully on any step failure so upload still returns.
        pipeline_summary = {"steps": []}
        try:
            analysis = service.analyze(dataset_id)
            pipeline_summary["steps"].append({
                "step": "profile",
                "status": "ok",
                "alerts": len(analysis.get("ydata_alerts", []))
            })
        except Exception as exc:
            pipeline_summary["steps"].append({"step": "profile", "status": "skipped", "reason": str(exc)})

        try:
            remediation = service.remediate(dataset_id)
            pipeline_summary["steps"].append({
                "step": "standardize",
                "status": "ok",
                "fixes": remediation.get("total_changes", 0),
                "failures": remediation.get("total_failures", 0),
            })
        except Exception as exc:
            pipeline_summary["steps"].append({"step": "standardize", "status": "skipped", "reason": str(exc)})

        # Guess target entity from the filename — user can override later
        fname_lower = filename.lower()
        target_entity = None
        if any(k in fname_lower for k in ("customer", "contact", "client")):
            target_entity = "Customers"
        elif any(k in fname_lower for k in ("vendor", "supplier")):
            target_entity = "Vendors"
        elif any(k in fname_lower for k in ("product", "sku", "catalog")):
            target_entity = "Products"
        elif any(k in fname_lower for k in ("invoice", "bill", "order", "receipt")):
            target_entity = "Invoices"

        # Wide-table heuristic: if a file has many columns, it\'s likely a flat
        # denormalized table containing multiple entities. Run schema normalization
        # and auto-route each detected entity into its master table.
        is_wide = len(df.columns) >= 10
        if is_wide and target_entity in (None, "Invoices"):
            # Treat as denormalized — run normalizer
            try:
                from core.normalizer import SchemaNormalizer
                normalizer = SchemaNormalizer()
                norm_result = normalizer.normalize(df, source_table_name=Path(filename).stem)
                norm_entities = norm_result.get("entities", [])

                ENTITY_TO_MASTER = {
                    "customers": "Customers", "customer": "Customers",
                    "vendors": "Vendors", "vendor": "Vendors", "suppliers": "Vendors",
                    "products": "Products", "product": "Products",
                    "orders": "Invoices", "order": "Invoices",
                    "invoices": "Invoices", "invoice": "Invoices", "bills": "Invoices",
                }
                ingested_per_entity = []
                skipped_entities = []
                for ent in norm_entities:
                    ent_name = ent.get("entity_name", "").lower().rstrip("s") + "s"
                    master_target = ENTITY_TO_MASTER.get(ent_name) or ENTITY_TO_MASTER.get(ent_name.rstrip("s"))
                    if not master_target:
                        skipped_entities.append(ent["entity_name"])
                        continue
                    # Build sub-DataFrame for this entity
                    ent_cols = ent.get("columns", [])
                    if not ent_cols:
                        continue
                    sub_df = df[ent_cols].drop_duplicates().reset_index(drop=True)
                    # Create a child dataset_id in the catalog for lineage
                    import uuid as _uuid
                    child_id = _uuid.uuid4().hex[:12]
                    ent_name_str = ent["entity_name"]
                    child_csv = service.raw_files_dir / f"{child_id}_v1.csv"
                    sub_df.to_csv(child_csv, index=False)
                    sub_source = f"{filename}::{ent_name_str}"
                    sub_fname = f"{Path(filename).stem}__{ent_name_str}.csv"
                    service.catalog.register_dataset(
                        source=sub_source,
                        filename=sub_fname,
                        rows=int(len(sub_df)),
                        columns=int(len(sub_df.columns)),
                        raw_csv_path=child_csv,
                    )
                    # Need real dataset_id from register_dataset (it generates its own)
                    real_child_id = service.catalog.db_path  # placeholder — we re-query
                    # Actually fetch the just-created child_id
                    import sqlite3 as _sql
                    with _sql.connect(service.catalog.db_path) as _c:
                        row = _c.execute(
                            "SELECT dataset_id FROM datasets WHERE source=? ORDER BY loaded_at DESC LIMIT 1",
                            (sub_source,)
                        ).fetchone()
                    real_child_id = row[0] if row else child_id
                    service._cache[real_child_id] = {
                        "dataframe": sub_df,
                        "source":    sub_source,
                        "loaded_at": pipeline_summary.get("loaded_at", ""),
                    }
                    # Ingest this entity into master
                    try:
                        ing = service.ingest_to_master(real_child_id, master_target)
                        ingested_per_entity.append({
                            "entity": ent["entity_name"],
                            "master": master_target,
                            "rows_inserted": ing.get("rows_inserted", 0),
                            "rows_duplicates_found": ing.get("rows_duplicates_found", 0),
                            "fields_enriched_total": ing.get("fields_enriched_total", 0),
                        })
                    except Exception as exc:
                        skipped_entities.append(f'{ent_name_str} (ingest error: {exc})')

                pipeline_summary["steps"].append({
                    "step": "auto_normalize",
                    "status": "ok",
                    "entities_detected": len(norm_entities),
                    "ingested": ingested_per_entity,
                    "skipped": skipped_entities,
                })
                # Don\'t also do the single-entity ingest path below
                target_entity = None
            except Exception as exc:
                pipeline_summary["steps"].append({
                    "step": "auto_normalize",
                    "status": "failed",
                    "reason": str(exc),
                })

        if target_entity:
            try:
                ingest = service.ingest_to_master(dataset_id, target_entity)
                pipeline_summary["steps"].append({
                    "step": "ingest",
                    "status": "ok",
                    "target_entity":          target_entity,
                    "rows_inserted":          ingest.get("rows_inserted", 0),
                    "rows_duplicates_found":  ingest.get("rows_duplicates_found", 0),
                    "rows_enriched":          ingest.get("rows_enriched", 0),
                    "fields_enriched_total":  ingest.get("fields_enriched_total", 0),
                    "rows_rejected":          ingest.get("rows_rejected", 0),
                    "total_in_master":        ingest.get("total_in_master", 0),
                    "duplicate_details":      ingest.get("duplicate_details", []),
                })
            except Exception as exc:
                pipeline_summary["steps"].append({
                    "step": "ingest", "status": "failed",
                    "target_entity": target_entity, "reason": str(exc),
                })
        else:
            pipeline_summary["steps"].append({
                "step": "ingest", "status": "skipped",
                "reason": "Could not auto-detect entity type from filename. Use the platform Ingest widget to route manually.",
            })

        return jsonify({
            "dataset_id":   dataset_id,
            "filename":     filename,
            "rows":         len(df),
            "columns":      list(df.columns),
            "preview":      df.head(5).fillna("").to_dict(orient="records"),
            "auto_pipeline": pipeline_summary,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/load-sqlite", methods=["POST"])
def load_sqlite():
    payload = request.get_json(force=True)
    db_path = payload.get("db_path")
    table = payload.get("table")
    if not db_path or not table:
        return jsonify({"error": "db_path and table are required"}), 400
    try:
        dataset_id, df = service.load_from_sqlite(db_path, table)
        return jsonify({
            "dataset_id": dataset_id,
            "table": table,
            "rows": len(df),
            "columns": list(df.columns),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─── Dataset operations ────────────────────────────────────────────────────

@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    return jsonify(service.list_datasets())


@app.route("/api/datasets/<dataset_id>", methods=["GET"])
def dataset_info(dataset_id: str):
    try:
        return jsonify(service.get_dataset_info(dataset_id))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/api/datasets/<dataset_id>/analyze", methods=["POST"])
def analyze_dataset(dataset_id: str):
    payload = request.get_json(silent=True) or {}
    table_name = payload.get("table_name", "data")
    try:
        result = service.analyze(dataset_id, table_name=table_name)
        # Don't return the internal _report object
        return jsonify(result)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<dataset_id>/remediate", methods=["POST"])
def remediate_dataset(dataset_id: str):
    payload = request.get_json(silent=True) or {}
    column_types = payload.get("column_types")
    try:
        result = service.remediate(dataset_id, column_types=column_types)
        return jsonify(result)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<dataset_id>/export", methods=["GET"])
def export_dataset(dataset_id: str):
    fmt = request.args.get("format", "csv")
    try:
        path = service.export(dataset_id, format=fmt)
        return send_file(path, as_attachment=True)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<dataset_id>/materialize", methods=["POST"])
def materialize_normalization(dataset_id: str):
    try:
        return jsonify(service.materialize_normalization(dataset_id))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<dataset_id>/normalized.zip", methods=["GET"])
def download_normalized_zip(dataset_id: str):
    zip_path = config.REPORTS_DIR / f"{dataset_id}_normalized.zip"
    if not zip_path.exists():
        return jsonify({"error": "Normalized files not found. Run /materialize first."}), 404
    return send_file(zip_path, as_attachment=True)


@app.route("/api/datasets/<dataset_id>/report", methods=["GET"])
def get_report(dataset_id: str):
    report_path = config.REPORTS_DIR / f"{dataset_id}_profile.html"
    if not report_path.exists():
        return jsonify({"error": "Report not found. Run /analyze first."}), 404
    return send_file(report_path)


# ─── Health ────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ollama_url": config.OLLAMA_URL,
        "ollama_model": config.OLLAMA_MODEL,
        "supported_formats": list(config.SUPPORTED_FORMATS),
    })




@app.route("/api/datasets/<dataset_id>/history", methods=["GET"])
def dataset_history(dataset_id: str):
    """Return versions, analyses, and remediations for a dataset."""
    try:
        return jsonify(service.get_history(dataset_id))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<dataset_id>/rollback", methods=["POST"])
def rollback_dataset(dataset_id: str):
    """Roll back to a previous version. Body: {"version_number": int}"""
    payload = request.get_json(silent=True) or {}
    version = payload.get("version_number")
    if not isinstance(version, int):
        return jsonify({"error": "version_number (int) is required"}), 400
    try:
        return jsonify(service.rollback(dataset_id, version))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/datasets/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id: str):
    """Delete a dataset and all related history."""
    try:
        ok = service.catalog.delete_dataset(dataset_id)
        service._cache.pop(dataset_id, None)
        return jsonify({"deleted": ok, "dataset_id": dataset_id})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500




@app.route("/api/resolve-entities", methods=["POST"])
def resolve_entities():
    """Cross-source entity resolution.

    Body:
      {
        "dataset_id_a": "abc123",
        "dataset_id_b": "def456",        // optional — if omitted, dedup within A
        "match_threshold": 0.7            // optional
      }
    """
    payload = request.get_json(silent=True) or {}
    a = payload.get("dataset_id_a")
    b = payload.get("dataset_id_b")
    threshold = float(payload.get("match_threshold", 0.7))
    if not a:
        return jsonify({"error": "dataset_id_a is required"}), 400
    try:
        return jsonify(service.resolve_entities(a, b, threshold))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500




@app.route("/api/find-duplicates-across-files", methods=["POST"])
def find_duplicates_across_files():
    """Find rows sharing identifier values across multiple datasets.

    Body:
      {
        "dataset_ids": ["id1", "id2", "id3", ...],
        "id_columns": ["bill_no", "invoice_ref"]   // optional — auto-detect if absent
      }
    """
    payload = request.get_json(silent=True) or {}
    dataset_ids = payload.get("dataset_ids") or []
    id_columns = payload.get("id_columns")
    detect_only = bool(payload.get("detect_only"))
    if not isinstance(dataset_ids, list) or len(dataset_ids) < 2:
        return jsonify({"error": "dataset_ids must be a list of at least 2 ids"}), 400
    try:
        if detect_only:
            # Return only the per-dataset detected ID columns, no dedup yet.
            # Used by the dashboard\'s two-step modal: step 1 picks files,
            # step 2 picks which column to match on.
            result = service.find_cross_file_duplicates(dataset_ids, id_columns=["__never_match__"])
            return jsonify({
                "datasets": result.get("datasets", []),
                "summary": {"total_clusters": 0, "total_duplicate_rows": 0, "rows_that_would_be_archived": 0},
                "duplicate_clusters": [],
                "id_columns_used": [],
            })
        return jsonify(service.find_cross_file_duplicates(dataset_ids, id_columns))
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500




@app.route("/api/archive-duplicates", methods=["POST"])
def archive_duplicates_endpoint():
    """Execute an archive plan from cross-file dedup.

    Body:
      {
        "archive_plan": [
          {
            "dataset_id": str,
            "row_indices": [int, ...],
            "id_column": str,
            "id_value": str,
            "related_dataset_id": str,
            "related_row_index": int
          }, ...
        ]
      }
    """
    payload = request.get_json(silent=True) or {}
    plan = payload.get("archive_plan") or []
    if not isinstance(plan, list) or not plan:
        return jsonify({"error": "archive_plan must be a non-empty list"}), 400
    try:
        return jsonify(service.archive_duplicate_rows(plan))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500




@app.route("/api/datasets/<dataset_id>/archived-rows", methods=["GET"])
def list_archived_rows(dataset_id: str):
    """Return all rows archived from a dataset (recoverable, not deleted)."""
    try:
        rows = service.catalog.list_archived_rows(dataset_id)
        return jsonify({
            "dataset_id": dataset_id,
            "count": len(rows),
            "archived_rows": rows,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500




@app.route("/api/suggest-dedup-columns", methods=["POST"])
def suggest_dedup_columns():
    """Recommend cross-file column matches for cross-file dedup.

    Body: {"dataset_ids": [str, str, ...]}
    Returns groups of columns that hold the same business concept across files.
    """
    payload = request.get_json(silent=True) or {}
    dataset_ids = payload.get("dataset_ids") or []
    if not isinstance(dataset_ids, list) or len(dataset_ids) < 2:
        return jsonify({"error": "dataset_ids must be a list of at least 2 ids"}), 400
    try:
        return jsonify(service.suggest_dedup_columns(dataset_ids))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500




@app.route("/api/platform/summary", methods=["GET"])
def platform_summary():
    return jsonify(service.get_platform_summary())


@app.route("/api/platform/ingest", methods=["POST"])
def platform_ingest():
    """Push a loaded dataset into a master entity.
    Body: {"dataset_id": str, "target_entity": "Customers"|"Vendors"|"Products"|"Invoices"}
    """
    payload = request.get_json(silent=True) or {}
    ds_id = payload.get("dataset_id")
    entity = payload.get("target_entity")
    if not ds_id or not entity:
        return jsonify({"error": "dataset_id and target_entity required"}), 400
    try:
        return jsonify(service.ingest_to_master(ds_id, entity))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/platform/master/<entity>", methods=["GET"])
def platform_master(entity: str):
    limit = int(request.args.get("limit", 50))
    return jsonify(service.list_master(entity, limit=limit))


@app.route("/api/platform/lineage", methods=["GET"])
def platform_lineage():
    limit = int(request.args.get("limit", 50))
    return jsonify(service.get_ingestion_log(limit=limit))


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)