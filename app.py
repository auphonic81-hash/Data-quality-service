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
        dataset_id, df = service.load_from_file(save_path)
        return jsonify({
            "dataset_id": dataset_id,
            "filename": filename,
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.head(5).fillna("").to_dict(orient="records"),
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
    if not isinstance(dataset_ids, list) or len(dataset_ids) < 2:
        return jsonify({"error": "dataset_ids must be a list of at least 2 ids"}), 400
    try:
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


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)