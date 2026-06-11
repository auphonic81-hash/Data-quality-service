"""Main service orchestrator.

DataQualityService is the public-facing entry point that ties together
ingestion, profiling, schema inference, quality detection, and remediation.
"""
from __future__ import annotations

import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .catalog import DatasetCatalog
from .consistency import RecordConsistencyChecker
from .ingestion import DataIngestion
from .cross_file_dedup import CrossFileDeduplicator
from .pdf_extraction import PDFExtractor
from .entity_resolution import EntityResolver
from .normalizer import SchemaNormalizer
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

        # Persistent catalog + on-disk storage for raw CSVs (survives restart)
        self.raw_files_dir = self.reports_dir / "raw"
        self.raw_files_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = DatasetCatalog(self.reports_dir / "catalog.sqlite3")

        self.ingestion = DataIngestion()
        self.profiler = DataProfiler()
        self.detector = QualityDetector()
        self.remediator = DataRemediator()
        self.schema_inferencer = SchemaInferencer(
            ollama_url=ollama_url, ollama_model=ollama_model
        )
        self.consistency_checker = RecordConsistencyChecker()
        self.normalizer = SchemaNormalizer()
        self.entity_resolver = EntityResolver()
        self.cross_file_dedup = CrossFileDeduplicator()
        self.pdf_extractor = PDFExtractor()

        self._cache: dict[str, dict[str, Any]] = {}

    # ─── Ingestion ─────────────────────────────────────────────────────────

    def load_from_file(self, file_path: str | Path) -> tuple[str, pd.DataFrame]:
        """Load a file, persist it on disk, and register it in the catalog.

        The catalog row + the on-disk CSV both survive server restarts.
        The in-memory cache is a performance layer only.
        """
        source_path = Path(file_path).resolve()
        df = self.ingestion.from_file(source_path)

        # Persist the raw CSV in our managed location so the catalog owns the file
        # (the user's uploads/ copy could be deleted/moved at any time)
        import shutil
        managed_csv = self.raw_files_dir / f"{source_path.stem}_{int(pd.Timestamp.utcnow().timestamp())}.csv"
        df.to_csv(managed_csv, index=False)

        dataset_id = self.catalog.register_dataset(
            source=str(source_path),
            filename=source_path.name,
            rows=int(len(df)),
            columns=int(len(df.columns)),
            raw_csv_path=managed_csv,
        )

        # Populate the in-memory cache for fast reads in the same session
        self._cache[dataset_id] = {
            "dataframe": df,
            "source": str(source_path),
            "loaded_at": pd.Timestamp.utcnow().isoformat() + "Z",
        }
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
        normalization = self.normalizer.normalize(df, source_table_name=table_name)

        analysis = {
            "dataset_id": dataset_id,
            "table_name": table_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "profile_summary": profile["ydata_summary"],
            "custom_findings": profile["custom_findings"],
            "quality_issues": quality,
            "consistency": consistency,
            "normalization": normalization,
            "inferred_schema": schema,
            "html_report_path": str(report_path),
            "analyzed_at": datetime.utcnow().isoformat() + "Z",
        }

        # Cache the analysis
        analysis = _to_json_safe(analysis)

        # Cache for fast access in the same session
        if dataset_id in self._cache:
            self._cache[dataset_id]["analysis"] = analysis

        # Persist in the catalog so history survives restarts
        catalog_row = self.catalog.get_dataset(dataset_id)
        current_version = catalog_row["current_version"] if catalog_row else 1
        self.catalog.record_analysis(
            dataset_id=dataset_id,
            version_number=current_version,
            payload=analysis,
            html_report_path=str(report_path),
        )

        return analysis

    @staticmethod
    def _extract_column_types_from_analysis(analysis: dict) -> dict[str, str]:
        """Pull {column_name: detected_type} out of a stored analysis result.

        The analyze step writes detected types into custom_findings.<col>.detected_type.
        Apply Fixes needs that map to know which column gets which remediator.
        """
        findings = analysis.get("custom_findings") or analysis.get("quality_issues", {})
        if not isinstance(findings, dict):
            return {}
        out: dict[str, str] = {}
        for col, info in findings.items():
            if isinstance(info, dict) and info.get("detected_type"):
                out[col] = info["detected_type"]
        return out

    def remediate(
        self, dataset_id: str, column_types: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Apply automated fixes. Creates a new on-disk version + audit row.

        The previous version remains untouched on disk so users can roll back.
        """
        catalog_row = self.catalog.get_dataset(dataset_id)
        if not catalog_row:
            raise KeyError(f"Dataset {dataset_id} not found")
        from_version = catalog_row["current_version"]

        df = self._get_dataset(dataset_id)
        before_preview = df.head(20).fillna("").astype(str).to_dict(orient="records")
        cleaned_df, change_log = self.remediator.remediate(df, column_types)
        after_preview = cleaned_df.head(20).fillna("").astype(str).to_dict(orient="records")

        total_changes = sum(log.get("changes", 0) for log in change_log.values())
        total_failures = sum(log.get("failures", 0) for log in change_log.values())

        # If nothing actually changed, don't create a noisy duplicate version.
        # Return immediately with a clear "no-op" result.
        if total_changes == 0:
            return _to_json_safe({
                "dataset_id": dataset_id,
                "from_version": from_version,
                "to_version": from_version,  # same version — no new one created
                "total_changes": 0,
                "total_failures": total_failures,
                "change_log": change_log,
                "no_changes": True,
                "message": "No fixes needed — data is already clean.",
                "columns": list(df.columns),
                "before_preview": before_preview,
                "after_preview": after_preview,
            })

        # Persist the cleaned data as a new immutable version on disk
        new_version_csv = self.raw_files_dir / f"{dataset_id}_v{from_version + 1}.csv"
        cleaned_df.to_csv(new_version_csv, index=False)

        new_version = self.catalog.add_version(
            dataset_id=dataset_id,
            csv_path=new_version_csv,
            change_summary=f"Apply Fixes: {total_changes} changes, {total_failures} failures",
        )

        self.catalog.record_remediation(
            dataset_id=dataset_id,
            from_version=from_version,
            to_version=new_version,
            change_log=change_log,
            total_changes=int(total_changes),
            total_failures=int(total_failures),
        )

        self._cache[dataset_id] = {
            "dataframe": cleaned_df,
            "source": catalog_row["source"],
            "loaded_at": catalog_row["loaded_at"],
        }

        return _to_json_safe({
            "dataset_id": dataset_id,
            "from_version": from_version,
            "to_version": new_version,
            "total_changes": total_changes,
            "total_failures": total_failures,
            "change_log": change_log,
            "cleaned_file_path": str(new_version_csv),
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
        """List all datasets from the persistent catalog (survives restarts)."""
        rows = self.catalog.list_datasets()
        return [
            {
                "dataset_id": r["dataset_id"],
                "source": r["source"],
                "filename": r["filename"],
                "rows": r["rows"],
                "columns": r["columns"],
                "loaded_at": r["loaded_at"],
                "current_version": r["current_version"],
                "remediated": r["current_version"] > 1,
            }
            for r in rows
        ]

    def get_dataset_info(self, dataset_id: str) -> dict[str, Any]:
        """Get metadata about a dataset, loading from disk if needed."""
        catalog_row = self.catalog.get_dataset(dataset_id)
        if not catalog_row:
            raise KeyError(f"Dataset {dataset_id} not found")
        df = self._get_dataset(dataset_id)
        return {
            "dataset_id": dataset_id,
            "source": catalog_row["source"],
            "filename": catalog_row["filename"],
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "loaded_at": catalog_row["loaded_at"],
            "current_version": catalog_row["current_version"],
            "remediated": catalog_row["current_version"] > 1,
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

    def materialize_normalization(self, dataset_id: str) -> dict[str, Any]:
            """Materialize the inferred normalized schema as physical CSV files.

            Returns a dict with:
              - tables: list of {name, rows, columns, csv_path}
              - zip_path: path to a ZIP containing all CSVs
            """
            df = self._get_dataset(dataset_id)
            norm = self.normalizer.normalize(df, source_table_name=dataset_id[:8])

            if not norm.get("is_denormalized"):
                return {
                    "materialized": False,
                    "reason": norm.get("skipped_reason", "No denormalization detected — nothing to split."),
                }

            materialized = self.normalizer.materialize(
                df,
                entities=norm["entities"],
                fact_columns=norm["fact_table_columns"],
                source_table_name=dataset_id[:8],
            )

            output_dir = self.reports_dir / f"{dataset_id}_normalized"
            output_dir.mkdir(parents=True, exist_ok=True)

            tables_info: list[dict[str, Any]] = []
            for name, table_df in materialized.items():
                csv_path = output_dir / f"{name}.csv"
                table_df.to_csv(csv_path, index=False)
                tables_info.append({
                    "name": name,
                    "rows": int(len(table_df)),
                    "columns": list(table_df.columns),
                    "preview": _to_json_safe(
                        table_df.head(5).fillna("").astype(str).to_dict(orient="records")
                    ),
                    "csv_path": str(csv_path),
                })

            # Bundle into a ZIP for download
            zip_path = self.reports_dir / f"{dataset_id}_normalized.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for info in tables_info:
                    zf.write(info["csv_path"], arcname=f"{info['name']}.csv")

            return _to_json_safe({
                "materialized": True,
                "dataset_id": dataset_id,
                "entity_count": len(norm["entities"]),
                "tables": tables_info,
                "zip_path": str(zip_path),
                "create_statements": norm["create_statements"],
            })

    def get_history(self, dataset_id: str) -> dict[str, Any]:
        """Return the full history for a dataset: versions, analyses, remediations."""
        catalog_row = self.catalog.get_dataset(dataset_id)
        if not catalog_row:
            raise KeyError(f"Dataset {dataset_id} not found")
        return _to_json_safe({
            "dataset_id": dataset_id,
            "filename": catalog_row["filename"],
            "current_version": catalog_row["current_version"],
            "loaded_at": catalog_row["loaded_at"],
            "versions": self.catalog.list_versions(dataset_id),
            "analyses": self.catalog.list_analyses(dataset_id),
            "remediations": self.catalog.list_remediations(dataset_id),
        })

    def rollback(self, dataset_id: str, version_number: int) -> dict[str, Any]:
        """Roll back to a previous version. The CSV on disk is unchanged
        (immutable history); we just point current_version at the older row
        and refresh the in-memory cache."""
        catalog_row = self.catalog.get_dataset(dataset_id)
        if not catalog_row:
            raise KeyError(f"Dataset {dataset_id} not found")

        ok = self.catalog.rollback_to_version(dataset_id, version_number)
        if not ok:
            raise ValueError(f"Version {version_number} does not exist for this dataset")

        # Drop the in-memory cache so the next read pulls from the rolled-back CSV
        self._cache.pop(dataset_id, None)

        return {
            "dataset_id": dataset_id,
            "rolled_back_to": version_number,
            "from_version": catalog_row["current_version"],
        }

    def resolve_entities(
        self,
        dataset_id_a: str,
        dataset_id_b: str | None = None,
        match_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Resolve entities. Two modes:
          - If dataset_id_b is provided → link_across_sources between A and B
          - If only dataset_id_a → dedup_within that dataset
        """
        df_a = self._get_dataset(dataset_id_a)
        info_a = self.catalog.get_dataset(dataset_id_a)
        name_a = info_a["filename"] if info_a else dataset_id_a

        if dataset_id_b:
            df_b = self._get_dataset(dataset_id_b)
            info_b = self.catalog.get_dataset(dataset_id_b)
            name_b = info_b["filename"] if info_b else dataset_id_b
            result = self.entity_resolver.link_across_sources(
                df_a, df_b,
                source_a_name=name_a, source_b_name=name_b,
                match_threshold=match_threshold,
            )
        else:
            result = self.entity_resolver.dedup_within(
                df_a, source_name=name_a, match_threshold=match_threshold,
            )

        return _to_json_safe(result)

    def suggest_dedup_columns(self, dataset_ids: list[str]) -> dict[str, Any]:
        """Recommend cross-file column matches for dedup."""
        bundles = []
        for ds_id in dataset_ids:
            catalog_row = self.catalog.get_dataset(ds_id)
            if not catalog_row:
                raise KeyError(f"Dataset {ds_id} not found")
            bundles.append({
                "dataset_id": ds_id,
                "name": catalog_row["filename"],
                "dataframe": self._get_dataset(ds_id),
            })
        groups = self.cross_file_dedup.suggest_column_groups(bundles)
        # Heuristic classification by default — deterministic, fast, reliable.
        # LLM re-analysis available via a separate endpoint if user requests it.
        groups = self.cross_file_dedup.classify_groups(groups)
        return _to_json_safe({"groups": groups})

    def find_cross_file_duplicates(
        self,
        dataset_ids: list[str],
        id_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect rows that share an identifier across multiple datasets."""
        if len(dataset_ids) < 2:
            return {
                "error": "Need at least 2 datasets",
                "datasets": [], "duplicate_clusters": [],
                "summary": {"total_clusters": 0, "total_duplicate_rows": 0, "rows_that_would_be_archived": 0},
            }

        bundles: list[dict[str, Any]] = []
        for ds_id in dataset_ids:
            catalog_row = self.catalog.get_dataset(ds_id)
            if not catalog_row:
                raise KeyError(f"Dataset {ds_id} not found")
            bundles.append({
                "dataset_id": ds_id,
                "name": catalog_row["filename"],
                "dataframe": self._get_dataset(ds_id),
            })

        result = self.cross_file_dedup.find_duplicates(bundles, id_columns=id_columns)
        return _to_json_safe(result)

    def archive_duplicate_rows(
        self,
        archive_plan: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute the archive plan from cross-file dedup.

        archive_plan is a list of:
          {
            "dataset_id":      str,   # dataset to remove rows from
            "row_indices":     [int], # which rows to archive
            "id_column":       str,   # for audit
            "id_value":        str,   # for audit
            "related_dataset_id": str, # which dataset kept the master copy
            "related_row_index":  int,
          }

        For each affected dataset, we:
          1. Save the rows to the archive table (with metadata)
          2. Create a NEW version of the dataset on disk with those rows removed
          3. Update the catalog so the active version points at the cleaned file
          4. Write a remediation audit row
        """
        # Group plan entries by dataset_id so we touch each file only once
        from collections import defaultdict
        per_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in archive_plan:
            per_dataset[entry["dataset_id"]].append(entry)

        result_per_dataset: list[dict[str, Any]] = []

        for dataset_id, entries in per_dataset.items():
            catalog_row = self.catalog.get_dataset(dataset_id)
            if not catalog_row:
                continue
            df = self._get_dataset(dataset_id)
            from_version = catalog_row["current_version"]

            # Collect row indices to archive + build archive records
            rows_to_archive: list[dict[str, Any]] = []
            indices_to_drop: set[int] = set()
            for entry in entries:
                for row_idx in entry.get("row_indices", []):
                    if row_idx < 0 or row_idx >= len(df):
                        continue
                    indices_to_drop.add(int(row_idx))
                    rows_to_archive.append({
                        "original_row_index": int(row_idx),
                        "row_data": df.iloc[row_idx].fillna("").astype(str).to_dict(),
                        "related_dataset_id": entry.get("related_dataset_id"),
                        "related_row_index": entry.get("related_row_index"),
                        "id_column": entry.get("id_column"),
                        "id_value": entry.get("id_value"),
                    })

            if not rows_to_archive:
                continue

            # Save the rows in the archive table
            archive_ids = self.catalog.archive_rows(
                dataset_id=dataset_id,
                rows=rows_to_archive,
                reason="cross_file_duplicate",
            )

            # Build a new DataFrame without the archived rows
            keep_mask = ~df.index.isin(indices_to_drop)
            new_df = df[keep_mask].reset_index(drop=True)

            # Persist as a new immutable version
            new_csv = self.raw_files_dir / f"{dataset_id}_v{from_version + 1}.csv"
            new_df.to_csv(new_csv, index=False)
            new_version = self.catalog.add_version(
                dataset_id=dataset_id,
                csv_path=new_csv,
                change_summary=f"Archived {len(rows_to_archive)} duplicate rows (cross-file dedup)",
            )

            # Audit row
            self.catalog.record_remediation(
                dataset_id=dataset_id,
                from_version=from_version,
                to_version=new_version,
                change_log={
                    "cross_file_dedup": {
                        "archived_count": len(rows_to_archive),
                        "archive_ids": archive_ids,
                    }
                },
                total_changes=len(rows_to_archive),
                total_failures=0,
            )

            # Refresh the in-memory cache to the new version
            self._cache[dataset_id] = {
                "dataframe": new_df,
                "source": catalog_row["source"],
                "loaded_at": catalog_row["loaded_at"],
            }

            result_per_dataset.append({
                "dataset_id": dataset_id,
                "filename": catalog_row["filename"],
                "archived_count": len(rows_to_archive),
                "from_version": from_version,
                "to_version": new_version,
                "remaining_rows": int(len(new_df)),
            })

        return _to_json_safe({
            "datasets_affected": result_per_dataset,
            "total_archived": sum(d["archived_count"] for d in result_per_dataset),
        })

    def load_from_pdf(self, pdf_path: str | Path) -> dict[str, Any]:
        """Extract data from a PDF and load it as a dataset.

        Returns the dataset_id (loaded into the catalog like any other file)
        plus the extraction metadata so the user can see what was found.
        """
        from pathlib import Path as _P
        pdf_path = _P(pdf_path)

        result = self.pdf_extractor.extract(pdf_path)
        if result.get("error"):
            raise ValueError(result["error"])

        # Build a DataFrame from the extraction.
        # If we have a table, merge in any key-value pairs (invoice_no, customer_id,
        # invoice_date, total) as additional columns on every row — so the header
        # context isn\'t lost when only the line items would be in the table.
        kv = result.get("key_value") or {}
        if result["tables"]:
            df = result["tables"][0].copy()
            for k, v in kv.items():
                if k not in df.columns:
                    df[k] = v
        elif kv:
            df = pd.DataFrame([kv])
        else:
            raise ValueError(
                "PDF contained no tables and no recognizable key-value pairs. "
                "The document may be too unstructured for automatic extraction."
            )

        # Persist the extracted data as CSV. Use a temp path while we ask the
        # catalog to mint a dataset_id, then rename to the final path and update
        # the catalog so its raw_csv_path points at the actual file.
        filename = pdf_path.stem + ".csv"
        import uuid as _uuid
        tmp_csv = self.raw_files_dir / f"_tmp_{_uuid.uuid4().hex[:8]}.csv"
        df.to_csv(tmp_csv, index=False)

        dataset_id = self.catalog.register_dataset(
            source=str(pdf_path),
            filename=filename,
            rows=int(len(df)),
            columns=int(len(df.columns)),
            raw_csv_path=tmp_csv,
        )

        # Rename temp file to canonical and update the catalog row to match
        final_csv = self.raw_files_dir / f"{dataset_id}_v1.csv"
        tmp_csv.rename(final_csv)

        self.catalog.update_raw_csv_path(dataset_id, final_csv)

        self._cache[dataset_id] = {
            "dataframe": df,
            "source":    str(pdf_path),
            "loaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        return _to_json_safe({
            "dataset_id":      dataset_id,
            "filename":        filename,
            "rows":            int(len(df)),
            "columns":         int(len(df.columns)),
            "extraction": {
                "strategy":        result["strategy"],
                "page_count":      result["page_count"],
                "ocr_confidence":  result["ocr_confidence"],
                "key_value":       result["key_value"],
                "warnings":        result["warnings"],
            },
        })

    def _get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Get a dataset's DataFrame. Loads from disk on cache miss."""
        # Fast path: already in memory
        meta = self._cache.get(dataset_id)
        if meta:
            return meta["dataframe"]

        # Cache miss → load from the catalog's current version on disk
        csv_path = self.catalog.get_current_csv_path(dataset_id)
        if not csv_path or not csv_path.exists():
            raise KeyError(f"Dataset {dataset_id} not found in catalog or on disk")

        df = self.ingestion.from_file(csv_path)
        catalog_row = self.catalog.get_dataset(dataset_id)
        self._cache[dataset_id] = {
            "dataframe": df,
            "source": catalog_row["source"] if catalog_row else str(csv_path),
            "loaded_at": catalog_row["loaded_at"] if catalog_row else pd.Timestamp.utcnow().isoformat() + "Z",
        }
        return df
    



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
