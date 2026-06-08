"""Main service orchestrator.

DataQualityService is the public-facing entry point that ties together
ingestion, profiling, schema inference, quality detection, and remediation.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .consistency import RecordConsistencyChecker
from .ingestion import DataIngestion
from .profiling import DataProfiler
from .quality_detection import QualityDetector
from .remediation import DataRemediator
from .schema_inference import SchemaInferencer


class DataQualityService:
    """High-level service that orchestrates the full data quality pipeline."""

    def __init__(
        self,
        reports_dir: Path,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3.5:latest",
    ):
        self.reports_dir = Path(reports_dir).resolve()
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.ingestion = DataIngestion()
        self.profiler = DataProfiler()
        self.detector = QualityDetector()
        self.remediator = DataRemediator()
        self.schema_inferencer = SchemaInferencer(
            ollama_url=ollama_url, ollama_model=ollama_model
        )
        self.consistency_checker = RecordConsistencyChecker()

        self._cache: dict[str, dict[str, Any]] = {}

    # ─── Ingestion ─────────────────────────────────────────────────────────

    def load_from_file(self, file_path: str | Path) -> tuple[str, pd.DataFrame]:
        """Load a file and cache it. Returns (dataset_id, dataframe)."""
        df = self.ingestion.from_file(file_path)
        dataset_id = self._cache_dataset(df, source=str(file_path))
        return dataset_id, df

    def load_from_sqlite(
        self, db_path: str | Path, table: str
    ) -> tuple[str, pd.DataFrame]:
        df = self.ingestion.from_sqlite(db_path, table)
        dataset_id = self._cache_dataset(df, source=f"{db_path}::{table}")
        return dataset_id, df

    # ─── Pipeline operations ───────────────────────────────────────────────

    def analyze(
        self, dataset_id: str, table_name: str = "data"
    ) -> dict[str, Any]:
        """Run the full analysis pipeline on a cached dataset."""
        df = self._get_dataset(dataset_id)

        profile = self.profiler.profile(df, title=f"{table_name} Quality Report")
        report_path = self.reports_dir / f"{dataset_id}_profile.html"
        self.profiler.save_report(profile, report_path)

        quality = self.detector.detect_all(df)
        schema = self.schema_inferencer.infer(df, table_name=table_name)
        consistency = self.consistency_checker.check(df)

        analysis = {
            "dataset_id": dataset_id,
            "table_name": table_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "profile_summary": profile["ydata_summary"],
            "custom_findings": profile["custom_findings"],
            "quality_issues": quality,
            "consistency": consistency,
            "inferred_schema": schema,
            "html_report_path": str(report_path),
            "analyzed_at": datetime.utcnow().isoformat() + "Z",
        }

        # Cache the analysis
        analysis = _to_json_safe(analysis)
        self._cache[dataset_id]["analysis"] = analysis
        return analysis

    def remediate(
        self, dataset_id: str, column_types: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Apply automated fixes to a cached dataset."""
        df = self._get_dataset(dataset_id)
        before_preview = df.head(20).fillna("").astype(str).to_dict(orient="records")
        cleaned_df, change_log = self.remediator.remediate(df, column_types)
        after_preview = cleaned_df.head(20).fillna("").astype(str).to_dict(orient="records")

        # Replace dataset with cleaned version
        self._cache[dataset_id]["dataframe"] = cleaned_df
        self._cache[dataset_id]["remediated"] = True

        # Save cleaned data
        cleaned_path = self.reports_dir / f"{dataset_id}_cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)

        total_changes = sum(
            log.get("changes", 0) for log in change_log.values()
        )
        total_failures = sum(
            log.get("failures", 0) for log in change_log.values()
        )

        return _to_json_safe({
            "dataset_id": dataset_id,
            "total_changes": total_changes,
            "total_failures": total_failures,
            "change_log": change_log,
            "cleaned_file_path": str(cleaned_path),
            "columns": list(df.columns),
            "before_preview": before_preview,
            "after_preview": after_preview,
        })

    def export(self, dataset_id: str, format: str = "csv") -> Path:
        """Export the current state of a dataset."""
        df = self._get_dataset(dataset_id)
        if format == "csv":
            path = self.reports_dir / f"{dataset_id}_export.csv"
            df.to_csv(path, index=False)
        elif format == "json":
            path = self.reports_dir / f"{dataset_id}_export.json"
            df.to_json(path, orient="records", force_ascii=False, indent=2)
        elif format == "parquet":
            path = self.reports_dir / f"{dataset_id}_export.parquet"
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        return path

    # ─── Dataset cache ─────────────────────────────────────────────────────

    def list_datasets(self) -> list[dict[str, Any]]:
        """List all cached datasets."""
        return [
            {
                "dataset_id": dataset_id,
                "source": meta["source"],
                "rows": len(meta["dataframe"]),
                "columns": len(meta["dataframe"].columns),
                "loaded_at": meta["loaded_at"],
                "remediated": meta.get("remediated", False),
            }
            for dataset_id, meta in self._cache.items()
        ]

    def get_dataset_info(self, dataset_id: str) -> dict[str, Any]:
        """Get metadata about a cached dataset."""
        meta = self._cache.get(dataset_id)
        if not meta:
            raise KeyError(f"Dataset {dataset_id} not found")
        df = meta["dataframe"]
        return {
            "dataset_id": dataset_id,
            "source": meta["source"],
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "loaded_at": meta["loaded_at"],
            "remediated": meta.get("remediated", False),
            "has_analysis": "analysis" in meta,
        }

    def _cache_dataset(self, df: pd.DataFrame, source: str) -> str:
        """Store a dataset in cache and return its ID."""
        dataset_id = uuid.uuid4().hex[:12]
        self._cache[dataset_id] = {
            "dataframe": df,
            "source": source,
            "loaded_at": datetime.utcnow().isoformat() + "Z",
        }
        return dataset_id

    def _get_dataset(self, dataset_id: str) -> pd.DataFrame:
        meta = self._cache.get(dataset_id)
        if not meta:
            raise KeyError(f"Dataset {dataset_id} not found")
        return meta["dataframe"]
    


def _to_json_safe(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj
