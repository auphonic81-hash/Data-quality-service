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

from datetime import datetime
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np

# ===========================================================================
# Entity classification helper (added by classifier wiring patch)
# ===========================================================================
def _classify_dataset(service, dataset_id, filename):
    """Run entity_classifier on a sample of the dataset's landing/staging data.

    Returns (entity, confidence, debug_info). Returns (None, 0, info) on
    low confidence or errors — caller should fall back to filename routing.
    """
    try:
        from core.entity_classifier import classify_entity
        import sqlite3 as _sql
        import pandas as _pd
        _c = _sql.connect(service.catalog.db_path, timeout=30)
        try:
            row = _c.execute(
                "SELECT source_system FROM datasets WHERE dataset_id=?",
                (dataset_id,)
            ).fetchone()
            if not row:
                return None, 0, {"reason": "dataset not in catalog"}
            source_system = row[0]
            # Try landing table first, then any staging table for this source
            for tbl in (f"landing_{source_system}",):
                try:
                    sample = _pd.read_sql_query(f"SELECT * FROM {tbl} LIMIT 200", _c)
                    if not sample.empty:
                        return classify_entity(sample, filename)
                except Exception:
                    continue
            return None, 0, {"reason": f"no landing table for {source_system}"}
        finally:
            _c.close()
    except Exception as e:
        return None, 0, {"reason": f"classifier error: {e}"}


def _route_by_filename(filename):
    """Fallback entity routing by filename keywords.

    NOTE: 'order' deliberately routes to SalesOrders (not Invoices) — fixes
    the long-standing bug where sales orders polluted master_invoices.
    """
    fname_lower = filename.lower()
    if any(k in fname_lower for k in ("customer", "contact", "client", "audit")):
        return "Customers"
    if any(k in fname_lower for k in ("vendor", "supplier")):
        return "Vendors"
    if any(k in fname_lower for k in ("product", "sku", "catalog", "quarterly")):
        return "Products"
    if any(k in fname_lower for k in ("invoice", "bill", "billing", "receipt")):
        return "Invoices"
    if any(k in fname_lower for k in ("order", "sales_order")):
        return "SalesOrders"
    if any(k in fname_lower for k in ("employee", "staff", "personnel", "hr")):
        return "Employees"
    if any(k in fname_lower for k in ("account",)):
        return "Accounts"
    if any(k in fname_lower for k in ("contract",)):
        return "Contracts"
    if any(k in fname_lower for k in ("department", "dept")):
        return "Departments"
    if any(k in fname_lower for k in ("inventor",)):
        return "Inventory"
    if any(k in fname_lower for k in ("location", "address", "site")):
        return "Locations"
    if any(k in fname_lower for k in ("opportunit",)):
        return "Opportunities"
    if any(k in fname_lower for k in ("payment",)):
        return "Payments"
    if any(k in fname_lower for k in ("shipment", "shipping")):
        return "Shipments"
    if any(k in fname_lower for k in ("legacy_billing",)):
        return "Invoices"
    if any(k in fname_lower for k in ("legacy_product",)):
        return "Products"
    if any(k in fname_lower for k in ("legacy_customer",)):
        return "Customers"
    return None
# ===========================================================================



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


# ---------------------------------------------------------------------------
# File disposition helper (added by file_disposition wiring patch)
# ---------------------------------------------------------------------------
def _safe_dispose(item, source_type, catalog_db_path, ingest_result=None,
                  target_entity=None, parse_error=None):
    """Decide what to do with a source file after an ingestion attempt.

    Returns one of: "deleted", "asided", "kept", or "skipped" (no path).

    Only EXTERNAL paths get touched — files inside the project's uploads/
    working directory are never auto-deleted (they're the system's own
    working copies and may be needed for re-export or audit).
    """
    src = item.get("path") if isinstance(item, dict) else None
    if not src:
        return "skipped"
    # Database / REST API sources have no file on disk
    if source_type in ("database", "restapi"):
        return "skipped"
    # Don't touch the project's internal uploads/ working dir
    try:
        from pathlib import Path as _P
        uploads_dir = (_P.home() / "data-quality-service" / "uploads").resolve()
        if _P(src).resolve().is_relative_to(uploads_dir):
            return "skipped"
    except Exception:
        pass
    try:
        from core.file_disposition import decide_disposition, move_to_aside
        if parse_error:
            return move_to_aside(
                source_path=src,
                catalog_db_path=catalog_db_path,
                reason_category="parse_error",
                reason_details=f"Exception during ingestion: {parse_error}",
            ).value
        return decide_disposition(
            source_path=src,
            catalog_db_path=catalog_db_path,
            ingest_result=ingest_result,
            target_entity=target_entity,
        ).value
    except Exception as e:
        print(f"[disposition] WARN: {e}")
        return "kept"
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# File hash enrichment helper (added by unified hash patcher)
# Used by every file-listing endpoint (folder, sharepoint, excel) so the
# UI gets consistent duplicate-detection data regardless of which source
# tab the user is browsing.
# ---------------------------------------------------------------------------
def _enrich_with_hashes(resp_dict, source_label):
    """Mutate `resp_dict['files']` in-place, adding hash + duplicate fields.

    Returns the same dict. Safe to call when files list is empty or the
    file_hash module is unavailable — logs the warning and returns dict
    unchanged.
    """
    if not isinstance(resp_dict, dict) or 'files' not in resp_dict:
        return resp_dict
    try:
        from core.file_hash import upsert_hash
    except Exception as e:
        print(f'[hash] import warn for {source_label}: {e}')
        return resp_dict
    catalog_db = service.catalog.db_path
    for _f in resp_dict.get('files', []):
        _p = _f.get('path')
        if not _p:
            continue
        try:
            _info = upsert_hash(catalog_db, _p, source_label=source_label)
        except Exception as e:
            print(f'[hash] enrich warn for {_p}: {e}')
            continue
        if 'error' in _info:
            continue
        _fh = _info.get('file_hash', '')
        _ch = _info.get('content_hash')
        _f['file_hash']            = _fh
        _f['file_hash_short']      = _fh[:12] if _fh else ''
        _f['content_hash']         = _ch
        _f['content_hash_short']   = _ch[:12] if _ch else None
        _f['is_duplicate']         = _info.get('is_duplicate', False)
        _f['is_content_duplicate'] = _info.get('is_content_duplicate', False)
        _f['duplicate_of']         = _info.get('duplicate_of', [])
        _f['content_duplicate_of'] = _info.get('content_duplicate_of', [])
        _f['first_seen_at']        = _info.get('first_seen_at')
        # Canonical = oldest copy. If is_duplicate is True, another copy
        # exists with an earlier first_seen_at, so this one is NOT canonical.
        _f['is_canonical'] = not _info.get('is_duplicate', False)
    return resp_dict
# ---------------------------------------------------------------------------

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

        # Entity routing: classifier first (data-driven), filename fallback.
        # The classifier inspects PK pattern + columns + status vocabulary on
        # actual data; filename keywords run only if classifier is unsure.
        classified, _conf, _dbg = _classify_dataset(service, dataset_id, filename)
        print(f"[ROUTING-1] file={filename!r} classified={classified!r} conf={_conf} dbg={_dbg}")
        if classified:
            target_entity = classified
        else:
            target_entity = _route_by_filename(filename)
            print(f"[ROUTING-1] fallback -> {target_entity!r}")
        print(f"[ROUTING-1] FINAL target_entity={target_entity!r}")

        # Wide-table heuristic: if a file has many columns, it\'s likely a flat
        # denormalized table containing multiple entities. Run schema normalization
        # and auto-route each detected entity into its master table.
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


@app.route("/api/platform/architecture", methods=["GET"])
def platform_architecture():
    """Return the 3-layer database architecture summary.

    Layer 1 (Landing): raw source data per file/endpoint/SQLite table.
    Layer 2 (Normalized): FD-split children of wide landing tables.
    Layer 3 (Master): cross-source consolidated golden records.
    """
    import sqlite3 as _sql
    db = service.catalog.db_path
    con = _sql.connect(db, timeout=30)
    try:
        def list_tables(prefix):
            rows = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
                (f"{prefix}%",)
            ).fetchall()
            out = []
            for (n,) in rows:
                row_count = con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]
                col_count = len(con.execute(f"PRAGMA table_info({n})").fetchall())
                out.append({"table": n, "rows": int(row_count), "columns": int(col_count)})
            return out
        landing = list_tables("landing_")
        staging = list_tables("stg_")
        normalized = list_tables("norm_")
        master = list_tables("master_")
        operational_names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'landing_%' AND name NOT LIKE 'stg_%' "
            "AND name NOT LIKE 'norm_%' AND name NOT LIKE 'master_%' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()]
        return jsonify({
            "layers": {
                "landing": {
                    "description": "Layer 1 — Landing: every source ingested into the DB as a raw table. Same as Informatica MDM landing tables.",
                    "count": len(landing),
                    "tables": landing,
                },
                "staging": {
                    "description": "Layer 2 — Staging: cleansed + canonicalized data per source × entity. One stg_<entity>_<source> per landing → master mapping.",
                    "count": len(staging),
                    "tables": staging,
                },
                "normalized": {
                    "description": "Layer 2 — Normalized: wide landing tables get split based on functional dependencies. Table count INCREASES here.",
                    "count": len(normalized),
                    "tables": normalized,
                },
                "master": {
                    "description": "Layer 3 — Master: cross-source consolidation into fixed golden record tables. One table per business domain.",
                    "count": len(master),
                    "tables": master,
                },
            },
            "operational": {
                "description": "Catalog, audit log, lineage events — supporting infrastructure.",
                "count": len(operational_names),
                "tables": operational_names,
            },
            "total_tables": len(landing) + len(staging) + len(normalized) + len(master) + len(operational_names),
        })
    finally:
        con.close()


@app.route("/api/match_queue/<entity>", methods=["GET"])
def match_queue(entity):
    """Run Splink between the staging tables for this entity, return ranked match pairs.

    Each pair gets a match_probability — the steward reviews and approves/rejects.
    Caches result in memory so we don\'t re-run Splink on every page load.
    """
    import sqlite3 as _sql, pandas as _pd
    threshold = float(request.args.get("threshold", "0.5"))

    db_path = service.catalog.db_path
    con = _sql.connect(db_path, timeout=30)
    try:
        # Find staging tables for this entity
        stg_tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type=\'table\' "
            "AND name LIKE ? ORDER BY name",
            (f"stg_{entity.lower()}__%",)
        ).fetchall()]
    finally:
        con.close()

    if len(stg_tables) < 2:
        return jsonify({
            "entity": entity, "pairs": [], "total_pairs": 0,
            "note": f"Need 2+ staging tables for {entity}. Found: {stg_tables}",
        })

    # For demo, pick the first two
    src_a, src_b = stg_tables[0], stg_tables[1]

    try:
        from core.dedup_splink import link_cross_source
        result = link_cross_source(db_path, src_a, src_b, entity, threshold=threshold)
    except Exception as exc:
        return jsonify({"entity": entity, "error": str(exc),
                        "sources": [src_a, src_b]}), 200

    # Format pairs for UI
    pairs = []
    for m in result.get("sample_matches", []):
        pairs.append({
            "probability": round(float(m.get("match_probability", 0)), 4),
            "source_a": src_a, "source_a_id": m.get("source_a_id"),
            "source_b": src_b, "source_b_id": m.get("source_b_id"),
            "details": {k: m.get(k, "") for k in m if k not in (
                "match_probability","source_a_id","source_b_id"
            )},
        })

    return jsonify({
        "entity": entity,
        "sources_compared": [src_a, src_b],
        "blocking_column": result.get("blocking_column"),
        "common_columns": result.get("common_comparison_columns"),
        "threshold": threshold,
        "total_pairs": result.get("matches_at_threshold", 0),
        "pairs_shown": len(pairs),
        "pairs": pairs,
        "engine": result.get("engine", "Splink"),
    })



@app.route("/api/match_decision", methods=["POST"])
def match_decision():
    """Record a steward's Approve/Reject decision on a match pair.
    Does NOT trigger a merge — that's the roadmap. This persists the decision
    with timestamp and steward for audit."""
    import sqlite3 as _sql
    data = request.get_json(silent=True) or {}
    required = ("entity", "pair_a", "pair_b", "decision")
    if not all(k in data for k in required):
        return jsonify({"error": f"missing fields, need: {required}"}), 400
    if data["decision"] not in ("approve", "reject"):
        return jsonify({"error": "decision must be 'approve' or 'reject'"}), 400
    con = _sql.connect(service.catalog.db_path, timeout=30)
    try:
        con.execute("""INSERT INTO match_decisions
            (entity, pair_a, pair_b, probability, decision) 
            VALUES (?, ?, ?, ?, ?)""",
            (data["entity"], data["pair_a"], data["pair_b"],
             float(data.get("probability", 0)), data["decision"])
        )
        con.commit()
    finally:
        con.close()
    return jsonify({"status": "recorded", "decision": data["decision"]})


@app.route("/api/schema_enhancement", methods=["GET"])
def schema_enhancement_summary():
    """Return what the platform has auto-enhanced based on observed data:
    indexes added, columns re-typed, constraints inferred."""
    from core.schema_enhancement import (
        discover_indexes, refine_column_types, infer_constraints
    )
    db = service.catalog.db_path
    # Count existing auto-added indexes
    import sqlite3 as _sql
    con = _sql.connect(db, timeout=30)
    try:
        idx_rows = con.execute(
            "SELECT tbl_name, name FROM sqlite_master WHERE type='index' "
            "AND name LIKE 'idx_%' AND sql IS NOT NULL ORDER BY tbl_name"
        ).fetchall()
    finally:
        con.close()
    refinements = refine_column_types(db)
    constraints = infer_constraints(db)
    # Group refinements by detected type
    by_type = {}
    for r in refinements:
        by_type.setdefault(r["actual_type"], 0)
        by_type[r["actual_type"]] += 1
    return jsonify({
        "indexes": {
            "total": len(idx_rows),
            "sample": [{"table": r[0], "name": r[1]} for r in idx_rows[:20]],
        },
        "column_type_refinements": {
            "total": len(refinements),
            "by_type": by_type,
            "sample": refinements[:15],
        },
        "constraints_inferred": {
            "total": len(constraints),
            "sample": constraints[:15],
        },
    })


@app.route("/api/schema_enhancement/run", methods=["POST"])
def schema_enhancement_run():
    """Re-run the full enhancement pass."""
    from core.schema_enhancement import run_full_enhancement
    return jsonify(run_full_enhancement(service.catalog.db_path))


@app.route("/api/platform/aside_files", methods=["GET"])
def platform_aside_files():
    """List files held ASIDE on disk — both unstructured docs (source-system
    layer, per Informatica architecture) AND the quarantine directory where
    the disposition layer moves files that fail classification or parsing.

    Quarantined files are enriched with their reason_category and
    reason_details from the file_quarantine table so the data steward
    knows exactly why each file was rejected.
    """
    from pathlib import Path as _P
    import sqlite3 as _sql_q

    # Pre-load quarantine metadata so we can enrich the disposition section
    quarantine_meta = {}
    try:
        _con_q = _sql_q.connect(service.catalog.db_path, timeout=30)
        try:
            _rows_q = _con_q.execute(
                "SELECT aside_path, original_path, reason_category, "
                "reason_details, moved_at FROM file_quarantine"
            ).fetchall()
            for ap, op, rc, rd, mv in _rows_q:
                quarantine_meta[ap] = {
                    "original_path":   op,
                    "reason_category": rc,
                    "reason_details":  rd,
                    "moved_at":        mv,
                }
        finally:
            _con_q.close()
    except Exception as _eq:
        print(f"[aside_files] quarantine lookup WARN: {_eq}")

    aside_dirs = [
        ("SharePoint Finance", _P("/home/alsubaihi/demo_sources/sharepoint_finance"), "source_layer"),
        ("Aside Letters & Memos",     _P("/home/alsubaihi/demo_sources/aside_letters"),     "source_layer"),
        ("Quarantine - Rejected by Validation",
         _P.home() / "data-quality-service" / "aside", "quarantine"),
    ]
    sections = []
    total_files = 0
    for label, d, section_type in aside_dirs:
        if not d.exists():
            continue
        files = []
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.name.startswith("."):
                continue
            # Hide .reason.txt sidecars — they're metadata for the file next to them
            if f.name.endswith(".reason.txt"):
                continue
            file_entry = {
                "name":      f.name,
                "size_kb":   round(f.stat().st_size / 1024, 1),
                "extension": f.suffix.lower(),
            }
            # For quarantine section, enrich with DB metadata
            if section_type == "quarantine":
                meta = quarantine_meta.get(str(f))
                if meta:
                    file_entry["reason_category"] = meta["reason_category"]
                    file_entry["reason_details"]  = meta["reason_details"]
                    file_entry["original_path"]   = meta["original_path"]
                    file_entry["moved_at"]        = meta["moved_at"]
            files.append(file_entry)
        sections.append({
            "label":        label,
            "directory":    str(d),
            "section_type": section_type,
            "file_count":   len(files),
            "files":        files,
        })
        total_files += len(files)

    return jsonify({
        "description": (
            "Two-tier aside: (1) source-system layer holds unstructured docs "
            "waiting for processing (Informatica model), (2) quarantine layer "
            "holds files the disposition system rejected, with reason audit trail."
        ),
        "total_aside_files": total_files,
        "sections":          sections,
    })


@app.route("/api/platform/normalize_landing", methods=["POST"])
def platform_normalize_landing():
    """Trigger Layer 2 — run schema normalization on landing tables."""
    from core.normalize_landing import normalize_all_landing_tables
    summary = normalize_all_landing_tables(service.catalog.db_path, min_columns=5)
    return jsonify(summary)


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
    q = (request.args.get("q", "") or "").strip().lower()
    # Fetch a broader pool if searching, else honor the limit directly
    fetch_limit = 20000 if q else limit
    result = service.list_master(entity, limit=fetch_limit)
    if q:
        rows = result.get("rows", []) if isinstance(result, dict) else result
        filtered = []
        for row in rows:
            for v in row.values():
                if v is not None and q in str(v).lower():
                    filtered.append(row)
                    break
            if len(filtered) >= limit:
                break
        if isinstance(result, dict):
            result["rows"] = filtered
        else:
            result = filtered
    return jsonify(result)


@app.route("/api/platform/lineage", methods=["GET"])
def platform_lineage():
    limit = int(request.args.get("limit", 50))
    return jsonify(service.get_ingestion_log(limit=limit))

@app.route("/api/platform/record_sources", methods=["GET"])
def platform_record_sources():
    """Per-record lineage: every source that contributed to a specific PK.
    Reads master_record_sources directly. Used by the drawer's timeline.
    """
    entity = request.args.get("entity", "")
    pk = request.args.get("pk", "")
    if not entity or not pk:
        return jsonify({"error": "entity and pk required"}), 400
    rows = service.master_repo.get_record_sources(entity, pk)
    return jsonify({"sources": rows})




# ═══════════════════════════════════════════════════════════════════════════
# CONNECT SOURCES WIZARD ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════
import sqlite3 as _sqlite3
from pathlib import Path as _Path

# Preset source folders — distinct directories per connector
LOCAL_FOLDER     = "/home/alsubaihi/demo_sources/local_folder"
SHAREPOINT_FOLDER = "/home/alsubaihi/demo_sources/sharepoint_finance"
EXCEL_FOLDER     = "/home/alsubaihi/demo_sources/excel_reports"
DATABASE_FILE    = "/home/alsubaihi/demo_dataset/legacy_finance.sqlite"
# Backward compat
DEMO_FOLDER      = LOCAL_FOLDER


@app.route("/api/sources/folder/list", methods=["GET"])
def sources_folder_list():
    """List ingestible files in the local demo folder."""
    folder = request.args.get("path", LOCAL_FOLDER)
    p = _Path(folder)
    if not p.exists():
        _resp = {"path": str(p), "files": [], "error": "Folder not found"}
        _enrich_with_hashes(_resp, 'folder')
        return jsonify(_resp), 404
    valid_ext = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv", ".pdf", ".txt", ".md", ".doc", ".docx"}
    files = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in valid_ext:
            stat = f.stat()
            files.append({
                "name":     f.name,
                "path":     str(f),
                "size_kb":  round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "type":     f.suffix.lower().lstrip("."),
            })
    _resp = {"path": str(p), "files": files}
    _enrich_with_hashes(_resp, 'folder')
    return jsonify(_resp)


@app.route("/api/sources/sharepoint/list", methods=["GET"])
def sources_sharepoint_list():
    """Same files, presented as a SharePoint Finance Department library."""
    folder = SHAREPOINT_FOLDER
    p = _Path(folder)
    if not p.exists():
        _resp = {"library": "Finance Department", "files": []}
        _enrich_with_hashes(_resp, 'sharepoint')
        return jsonify(_resp), 404
    valid_ext = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv", ".pdf", ".txt", ".md", ".doc", ".docx"}
    files = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in valid_ext:
            stat = f.stat()
            files.append({
                "name":         f.name,
                "path":         str(f),
                "size_kb":      round(stat.st_size / 1024, 1),
                "modified_by":  "Finance Dept",
                "synced_at":    datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "library":      "Finance Department",
                "type":         f.suffix.lower().lstrip("."),
            })
    _resp = {
        "library": "SharePoint - Finance Department",
        "tenant":  "company.sharepoint.com",
        "files":   files,
    }
    _enrich_with_hashes(_resp, 'sharepoint')
    return jsonify(_resp)




@app.route("/api/sources/excel/list", methods=["GET"])
def sources_excel_list():
    """List Excel files in the dedicated Excel reports directory."""
    folder = EXCEL_FOLDER
    p = _Path(folder)
    if not p.exists():
        _resp = {"library": "Excel Reports", "files": [], "error": "Folder not found"}
        _enrich_with_hashes(_resp, 'excel')
        return jsonify(_resp), 404
    files = []
    for f in sorted(p.iterdir()):
        if f.is_file() and f.suffix.lower() in (".xlsx", ".xls"):
            stat = f.stat()
            files.append({
                "name":      f.name,
                "path":      str(f),
                "size_kb":   round(stat.st_size / 1024, 1),
                "modified":  datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                "type":      f.suffix.lower().lstrip("."),
                "department": "Finance & Operations",
            })
    _resp = {"library": "Excel Reports", "folder": str(p), "files": files}
    _enrich_with_hashes(_resp, 'excel')
    return jsonify(_resp)




# ─── REST API connector — mock HR system on port 5005 ───
REST_API_BASE = "http://localhost:5005"
CRM_API_BASE  = "http://localhost:5006"

@app.route("/api/sources/restapi/list", methods=["GET"])
def sources_restapi_list():
    """Discover endpoints from connected REST API services (HR + CRM)."""
    import requests as _requests
    SKIP_ENDPOINTS = {"departments", "leads"}
    services = [
        {"base": REST_API_BASE, "label": "Mock HR System"},
        {"base": CRM_API_BASE,  "label": "Mock CRM System"},
    ]
    endpoints = []
    services_meta = []
    for svc in services:
        try:
            base = svc["base"]
            h = _requests.get(f"{base}/api/v1/health", timeout=3).json()
            s = _requests.get(f"{base}/api/v1/schema", timeout=3).json()
            services_meta.append({
                "service": h.get("service", svc["label"]),
                "version": h.get("version", "?"),
                "base_url": svc["base"],
                "status":  "ok",
            })
            for ep in s.get("endpoints", []):
                if ep.get("entity", "").lower() in SKIP_ENDPOINTS:
                    continue
                endpoints.append({
                    "name":         ep["entity"],
                    "path":         ep["path"],
                    "base_url":     svc["base"],
                    "type":         "rest_endpoint",
                    "fields":       ep.get("fields", []),
                    "field_count":  len(ep.get("fields", [])),
                    "department":   svc["label"],
                    "size_kb":      None,
                    "modified":     "Live API",
                })
        except Exception as e:
            services_meta.append({
                "service": svc["label"],
                "status":  "unreachable",
                "error":   str(e),
                "base_url": svc["base"],
            })
    return jsonify({
        "library":   "REST APIs — Multiple SaaS Systems",
        "services":  services_meta,
        "files":     endpoints,
    })


def _fetch_rest_endpoint(path):
    """Pull all data from a REST endpoint, handling pagination if present."""
    import requests as _requests
    import pandas as _pd
    r = _requests.get(f"{REST_API_BASE}{path}?limit=2000&offset=0", timeout=10).json()
    rows = r.get("data", r) if isinstance(r, dict) else r
    return _pd.DataFrame(rows)



@app.route("/api/sources/database/tables", methods=["GET"])
def sources_database_tables():
    """List tables in a SQLite database (mimics ERP/SAP table list)."""
    db_path = request.args.get("path", DATABASE_FILE)
    if not _Path(db_path).exists():
        return jsonify({"database": db_path, "tables": [], "error": "Database file not found"}), 404
    try:
        with _sqlite3.connect(db_path, timeout=30) as con:
            tables = []
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
                tname = row[0]
                count = con.execute(f"SELECT COUNT(*) FROM \"{tname}\"").fetchone()[0]
                cols = [c[1] for c in con.execute(f"PRAGMA table_info(\"{tname}\")")]
                tables.append({
                    "name":    tname,
                    "rows":    count,
                    "columns": cols,
                })
            return jsonify({
                "database": db_path,
                "type":     "SQLite (acts as ERP/Legacy DB)",
                "tables":   tables,
            })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sources/batch_ingest", methods=["POST"])
def sources_batch_ingest():
    """Ingest a list of files or DB tables in one batch.

    Body:
      {"source_type": "folder"|"sharepoint"|"database",
       "items":       [{"path": "...", "type": "csv|pdf|table", "name": "..."}, ...]
      }
    Returns per-item results with rich pipeline summaries.
    """
    import pandas as _pd
    payload = request.get_json(silent=True) or {}
    source_type = payload.get("source_type")
    items = payload.get("items", [])
    if not items:
        return jsonify({"error": "No items selected"}), 400
    results = []
    for item in items:
        item_name = item.get("name", item.get("path", "unknown"))
        # --- pre-ingestion content_hash duplicate check ---
        # Catches semantic duplicates (same data, different file format /
        # different column order) BEFORE we touch master. PDFs and other
        # unstructured files have content_hash=None and fall through to
        # normal PK-based row dedup.
        try:
            from core.dedup_check import check_pre_ingestion_duplicate
            _dedup_result = check_pre_ingestion_duplicate(
                file_path=item.get("path"),
                source_type=source_type,
                catalog_db_path=service.catalog.db_path,
            )
        except Exception as _dedup_err:
            print(f"[dedup] check failed for {item_name}: {_dedup_err}")
            _dedup_result = None
        if _dedup_result:
            # Skip the ingestion entirely - file has been soft-deleted to duplicates/
            results.append({
                "item":               item_name,
                "status":             "duplicate_skipped",
                "reason":             _dedup_result.get("reason", "content matches existing file"),
                "canonical":          _dedup_result.get("canonical"),
                "content_hash":       _dedup_result.get("content_hash_short"),
                "disposition":        _dedup_result.get("disposition"),
            })
            continue
        # --- end pre-ingestion duplicate check ---
        try:
            if source_type == "database":
                # Pull table from SQLite into a temp CSV, then load it
                db_path = item.get("db_path") or DATABASE_FILE
                table_name = item.get("name")
                with _sqlite3.connect(db_path, timeout=30) as con:
                    df = _pd.read_sql_query(f'SELECT * FROM "{table_name}"', con)
                # Save as CSV in uploads, then run through the normal load path
                temp_csv = config.UPLOADS_DIR / f"db_{table_name}.csv"
                df.to_csv(temp_csv, index=False)
                ingestion_filename = f"{table_name}.csv"  # SQLite table treated as CSV
                save_path = temp_csv
            elif source_type == "restapi":
                # Pull data from a REST API endpoint into a temp CSV
                import requests as _requests
                import csv as _csv
                ep_path = item.get("path", "") or item.get("endpoint", "")
                ep_name = item.get("name", "endpoint")
                # Use the endpoint's own base_url (HR=5005, CRM=5006), fallback to HR
                ep_base = (item.get("base_url") or REST_API_BASE).rstrip("/")
                # Normalize path: ensure exactly one leading slash
                if ep_path and not ep_path.startswith("/"):
                    ep_path = "/" + ep_path
                full_url = f"{ep_base}{ep_path}?limit=5000&offset=0"
                resp = _requests.get(full_url, timeout=15)
                if resp.status_code != 200:
                    raise ValueError(f"REST endpoint {full_url} returned HTTP {resp.status_code}")
                try:
                    r = resp.json()
                except Exception as _exc:
                    raise ValueError(f"REST endpoint {full_url} returned non-JSON: {resp.text[:200]}")
                rows = r.get("data", r) if isinstance(r, dict) else r
                df = _pd.DataFrame(rows)
                if df.empty:
                    raise ValueError(f"REST endpoint {ep_path} returned no data")
                temp_csv = config.UPLOADS_DIR / f"api_{ep_name}.csv"
                df.astype(str).replace("nan", "", regex=False).to_csv(
                    temp_csv, index=False, quoting=_csv.QUOTE_NONNUMERIC
                )
                ingestion_filename = f"{ep_name}.csv"
                save_path = temp_csv
            else:
                # File path
                save_path = _Path(item["path"])
                ingestion_filename = save_path.name
                # Copy to uploads dir for catalog purposes
                import shutil as _shutil
                target = config.UPLOADS_DIR / ingestion_filename
                if save_path != target:
                    _shutil.copy(save_path, target)
                save_path = target

            # Now run the full pipeline (mimics what /api/upload does)
            ext = save_path.suffix.lower()
            if ext == ".pdf":
                pdf_result = service.load_from_pdf(save_path)
                dataset_id = pdf_result["dataset_id"]
                df_loaded = _pd.read_csv(service.catalog.get_dataset(dataset_id)["raw_csv_path"])
            else:
                _nm = item.get('name', save_path.stem); _ss = ('api_' + (_nm[4:] if _nm.startswith('api_') else _nm)) if source_type == 'restapi' else (('sqlite_' + (_nm[3:] if _nm.startswith('db_') else _nm)) if source_type == 'database' else None)
                dataset_id, df_loaded = service.load_from_file(save_path, source_system=_ss)

            # Cleansing already happened inside load_from_file via the auto_enhance hook.
            # The old service.remediate() call here was redundant AND it stripped column
            # names on re-read from disk — which is what caused 14K rows with no customer_id.

            # Entity routing: classifier first (data-driven), filename fallback.
            # Replaces the old filename-only chain that mis-routed orders to Invoices.
            classified, _conf, _dbg = _classify_dataset(service, dataset_id, ingestion_filename)
            print(f"[ROUTING-2] file={ingestion_filename!r} classified={classified!r} conf={_conf} dbg={_dbg}")
            # PK sanity check: if classified entity's PK column doesn't exist in the incoming
            # data (via any alias), the classification is wrong. Reject and fall through.
            if classified:
                try:
                    from core.master_repository import MASTER_TABLES as _MT
                    if classified in _MT:
                        _mt, _pk, _fmap = _MT[classified]
                        _c_pk = _sql.connect(service.catalog.db_path, timeout=30) if False else None
                        # Read landing table columns to check
                        import sqlite3 as _sqlc
                        _cx = _sqlc.connect(service.catalog.db_path, timeout=30)
                        try:
                            _rx = _cx.execute("SELECT source_system FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
                            _ssx = _rx[0] if _rx and _rx[0] else f"file_{save_path.stem}"
                            _landx = f"landing_{_ssx}"
                            _colsx = [r[1].lower() for r in _cx.execute(f"PRAGMA table_info({_landx})").fetchall()]
                        finally:
                            _cx.close()
                        # Strict PK check: the CANONICAL PK column name itself must be present.
                        # Aliases are for downstream mapping, not classification validation.
                        # Exception: certain safe aliases that always mean "this IS the PK" (e.g., 'id', 'code')
                        _pk_lower = _pk.lower()
                        _canonical_present = _pk_lower in _colsx
                        if not _canonical_present:
                            print(f"[ROUTING-2] canonical PK '{_pk}' not in columns {_colsx[:10]} - rejecting classification")
                            classified = None
                        else:
                            # Also verify uniqueness so we don't accept a non-unique column
                            try:
                                _cu = _sqlc.connect(service.catalog.db_path, timeout=30)
                                _n_total = _cu.execute(f"SELECT COUNT(*) FROM {_landx}").fetchone()[0]
                                _n_uniq = _cu.execute(f"SELECT COUNT(DISTINCT \"{_pk_lower}\") FROM {_landx}").fetchone()[0]
                                _cu.close()
                                _pk_unique = (_n_uniq / max(1, _n_total)) >= 0.9
                            except Exception:
                                _pk_unique = False
                            if not _pk_unique:
                                print(f"[ROUTING-2] canonical PK '{_pk}' present but not unique - rejecting classification")
                                classified = None
                            else:
                                print(f"[ROUTING-2] canonical PK '{_pk}' present and unique - keeping classification")
                except Exception as _e:
                    print(f"[ROUTING-2] PK check error: {_e}")
            if classified:
                target_entity = classified
            else:
                target_entity = _route_by_filename(ingestion_filename)
                print(f"[ROUTING-2] fallback -> {target_entity!r}")
            print(f"[ROUTING-2] FINAL target_entity={target_entity!r}")

            if target_entity:
                # Layer 1: write landing table before promoting to master
                try:
                    from core.landing import persist_to_landing
                    import sqlite3 as _sql
                    _c = _sql.connect(service.catalog.db_path, timeout=30)
                    try:
                        _r = _c.execute(
                            "SELECT source_system FROM datasets WHERE dataset_id=?",
                            (dataset_id,)
                        ).fetchone()
                        source_system = _r[0] if _r and _r[0] else ("sqlite" if source_type == "database" else f"file_{save_path.stem}")
                    finally:
                        _c.close()
                    persist_to_landing(
                        service.catalog.db_path,
                        f"db_{ingestion_filename}" if source_type == "database" else ingestion_filename,
                        df_loaded,
                        source_system
                    )
                except Exception as _le:
                    print(f"[landing] WARN: {_le}")
                try:
                    from core.staging import build_staging_table
                    import re as _re
                    _ln = f"db_{ingestion_filename}" if source_type == "database" else ingestion_filename
                    _ln = _re.sub(r'\.(csv|xlsx?|pdf)$', '', _ln.lower())
                    if source_type == "database":
                        _ln = f"landing_sqlite_{_ln[3:]}"
                    elif source_type == "restapi":
                        _ln = f"landing_api_{_ln[4:] if _ln.startswith('api_') else _ln}"
                    elif ingestion_filename.lower().endswith(".pdf") or (source_system or "").startswith("pdf_"):
                        # PDFs persist to landing_pdf_<stem>, not landing_file_<stem>
                        _ln = f"landing_pdf_{_ln}"
                    else:
                        _ln = f"landing_file_{_ln}"
                    _stg = build_staging_table(service.catalog.db_path, _ln)
                    print(f"[staging] {_ln}: {_stg.get('status')}")
                except Exception as _se:
                    print(f"[staging] WARN: {_se}")

                ing = service.ingest_to_master(dataset_id, target_entity)
                _disposition = _safe_dispose(
                    item, source_type, service.catalog.db_path,
                    ingest_result=ing, target_entity=target_entity,
                )
                results.append({
                    "item": ingestion_filename, "status": "ok", "mode": "direct",
                    "entity": target_entity,
                    "new":        ing.get("rows_inserted", 0),
                    "duplicates": ing.get("rows_duplicates_found", 0),
                    "enriched":   ing.get("fields_enriched_total", 0),
                    "total_in_master": ing.get("total_in_master", 0),
                    "disposition": _disposition,
                })
            else:
                # Auto-onboarding path: try proposer before asiding
                print(f"[AUTO-ONBOARD] ENTRY - starting proposer flow for {ingestion_filename!r}")
                _auto_created = None
                try:
                    from core.entity_proposer import EntityProposer
                    from core.master_repository import MASTER_TABLES as _MT
                    import sqlite3 as _sqlp
                    _cp = _sqlp.connect(service.catalog.db_path, timeout=30)
                    try:
                        _rp = _cp.execute(
                            "SELECT source_system FROM datasets WHERE dataset_id=?",
                            (dataset_id,)).fetchone()
                        _ss = _rp[0] if _rp and _rp[0] else f"file_{save_path.stem}"
                        _land = f"landing_{_ss}"
                        _sample_df = _pd.read_sql_query(
                            f"SELECT * FROM {_land} LIMIT 200", _cp)
                    finally:
                        _cp.close()
                    if not _sample_df.empty:
                        _prop = EntityProposer(service.catalog.db_path)
                        _pres = _prop.propose(_sample_df, filename=ingestion_filename,
                                              existing_entities={k: v for k, v in _MT.items()})
                        if _pres.get("status") == "proposed":
                            _reg = _prop.create_and_register(
                                _pres["proposal"],
                                created_from_file=ingestion_filename)
                            _auto_created = _reg
                            print(f"[AUTO-ONBOARD] created {_reg['master_table']} from {ingestion_filename}")
                        elif _pres.get("status") == "route_to_existing":
                            # Same schema as a previously-onboarded entity — reload it into MASTER_TABLES
                            # and route this ingest to it
                            _existing_ent = _pres["route_to_entity"]
                            print(f"[AUTO-ONBOARD] routing to existing entity: {_existing_ent}")
                            _all = _prop.load_all()
                            _match = next((e for e in _all if e["entity_name"] == _existing_ent), None)
                            if _match:
                                _MT[_existing_ent] = (_match["master_table"], _match["primary_key"],
                                                      _match["field_map"])
                                # Signal caller to route ingest via normal path
                                _auto_created = {
                                    "entity_name": _existing_ent,
                                    "master_table": _match["master_table"],
                                    "primary_key": _match["primary_key"],
                                    "columns_created": list(_match["field_map"].keys()),
                                    "_is_reroute": True,
                                }
                        else:
                            print(f"[AUTO-ONBOARD] proposer rejected: {_pres.get('reason')}")
                except Exception as _ape:
                    print(f"[AUTO-ONBOARD] error: {_ape}")

                if _auto_created:
                    # Register new entity in MASTER_TABLES runtime dict so
                    # ingest_to_master recognizes it. Uses simple identity field_map.
                    try:
                        from core.master_repository import MASTER_TABLES as _MTR
                        _new_ent = _auto_created["entity_name"]
                        _new_pk = _auto_created["primary_key"]
                        _new_tbl = _auto_created["master_table"]
                        # Build identity field_map: canonical → [canonical] for each column
                        _new_fmap = {c: [c] for c in _auto_created["columns_created"]}
                        _MTR[_new_ent] = (_new_tbl, _new_pk, _new_fmap)
                        print(f"[AUTO-ONBOARD] registered {_new_ent} in MASTER_TABLES")
                    except Exception as _re:
                        print(f"[AUTO-ONBOARD] register error: {_re}")

                    # Now populate the newly-created master by calling ingest_to_master
                    _pop_new = 0
                    _pop_dups = 0
                    try:
                        _ing2 = service.ingest_to_master(dataset_id, _new_ent)
                        _pop_new = _ing2.get("rows_inserted", 0)
                        _pop_dups = _ing2.get("rows_duplicates_found", 0)
                        print(f"[AUTO-ONBOARD] populated {_new_tbl}: +{_pop_new} rows")
                        _disposition = _safe_dispose(
                            item, source_type, service.catalog.db_path,
                            ingest_result=_ing2, target_entity=_new_ent,
                        )
                    except Exception as _pe:
                        print(f"[AUTO-ONBOARD] populate error: {_pe}")
                        _disposition = _safe_dispose(
                            item, source_type, service.catalog.db_path,
                            target_entity=_new_ent,
                        )

                    results.append({
                        "item": ingestion_filename,
                        "status": "ok",
                        "mode": "auto_onboarded",
                        "entity": _auto_created["entity_name"],
                        "master_table": _auto_created["master_table"],
                        "primary_key": _auto_created["primary_key"],
                        "columns_created": _auto_created["columns_created"],
                        "new": _pop_new,
                        "duplicates": _pop_dups,
                        "disposition": _disposition,
                        "note": f"New entity '{_new_ent}' auto-discovered and populated ({_pop_new} rows).",
                    })
                else:
                    _disposition = _safe_dispose(
                        item, source_type, service.catalog.db_path,
                        target_entity=None,
                    )
                    results.append({
                        "item": ingestion_filename,
                        "status": "asided" if _disposition == "asided" else "skipped",
                        "reason": "unrecognized_entity",
                        "disposition": _disposition,
                    })
        except Exception as exc:
            _msg = str(exc)
            _is_fmt = "Unsupported format" in _msg
            _reason = "unsupported_format" if _is_fmt else _msg
            _disposition = _safe_dispose(
                item, source_type, service.catalog.db_path,
                parse_error=_reason,
            )
            _entry = {
                "item": item_name,
                "status": "asided" if _is_fmt else "error",
                "reason": _reason if _is_fmt else None,
                "disposition": _disposition,
            }
            if not _is_fmt:
                _entry["error"] = _msg
            else:
                _entry.pop("reason", None) if _entry["reason"] is None else None
            results.append(_entry)

    # Pull final master summary for the wizard to display
    summary = service.get_platform_summary()
    return jsonify({"results": results, "summary": summary})




@app.route("/api/openlineage", methods=["GET"])
def openlineage_export():
    """Export lineage events in OpenLineage standard format.

    Compatible with Marquez, DataHub, or any OpenLineage-compatible
    governance system. Standard schema URL:
    https://openlineage.io/spec/2-0-2/OpenLineage.json
    """
    limit = int(request.args.get("limit", 200))
    emitter = service.master_repo.openlineage
    return jsonify({
        "events": emitter.export_all(limit),
        "total":  emitter.count(),
        "schema": "https://openlineage.io/spec/2-0-2/OpenLineage.json",
    })




@app.route("/api/platform/full_reset", methods=["POST"])
def platform_full_reset():
    """Wipe ALL data tables for a clean live demo.
    Keeps table structures but drops all data rows.
    Also drops auto-promoted master entities so they re-appear during demo.
    """
    import sqlite3 as _sql
    db = service.catalog.db_path
    con = _sql.connect(db, timeout=30)
    report = {}
    try:
        # Find all tables to wipe
        all_tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()]

        # Tables to DROP completely (auto-promoted ones, will be recreated on ingest)
        auto_promoted = [t for t in all_tables if t.startswith("master_") and t not in (
            "master_customers","master_vendors","master_products",
            "master_invoices","master_employees","master_record_sources"
        )]
        # Tables to CLEAR (keep structure, wipe rows)
        clear_tables = [
            "master_customers","master_vendors","master_products",
            "master_invoices","master_employees","master_record_sources",
            "ingestion_log","datasets",
        ] + [t for t in all_tables if t.startswith("landing_")]            + [t for t in all_tables if t.startswith("stg_")]                 + [t for t in all_tables if t.startswith("norm_")]

        # Drop auto-promoted master tables
        for tbl in auto_promoted:
            con.execute(f"DROP TABLE IF EXISTS {tbl}")
            report[f"dropped_{tbl}"] = True

        # Clear core tables
        for tbl in clear_tables:
            if tbl in all_tables:
                count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                con.execute(f"DELETE FROM {tbl}")
                report[tbl] = int(count)

        con.commit()
    finally:
        con.close()

    # Also clear service cache
    service._cache.clear()

    return jsonify({"full_reset": True, "wiped": report})


@app.route("/api/platform/reset", methods=["POST"])
def platform_reset():
    """Wipe master tables, ingestion log, archived rows for a fresh state.

    Used to demonstrate the platform on a clean slate — the engine, schemas,
    and source connectors stay untouched; only the working data is cleared.
    """
    import sqlite3 as _sqlite3
    tables_to_wipe = [
        "master_customers", "master_vendors", "master_products", "master_invoices", "master_employees",
        "ingestion_log", "openlineage_events", "archived_rows",
    ]
    wiped = {}
    with _sqlite3.connect(service.catalog.db_path, timeout=30) as con:
        for t in tables_to_wipe:
            try:
                before = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                con.execute(f"DELETE FROM {t}")
                wiped[t] = before
            except _sqlite3.OperationalError:
                wiped[t] = None  # table doesn\'t exist
        con.commit()
    return jsonify({"reset": True, "wiped": wiped})


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=False, use_reloader=False)