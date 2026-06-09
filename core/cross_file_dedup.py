"""Cross-file deterministic deduplication.

Finds rows that share an identifier value across multiple datasets.

Unlike fuzzy entity resolution (which guesses based on names/emails/phones),
this is DETERMINISTIC: if column "Invoice_Reference" exists in two files and
two rows share the same value, they ARE the same record. No probability needed.

The challenge is detecting WHICH columns are identifiers — clients use many
naming conventions: bill_no, invoice_ref, sku, customer_id, plate_number,
emirates_id, order_number, etc.

Detection uses three signals (any one triggers a candidacy):
  1. NAME signal:   column name contains a known identifier token
  2. UNIQUENESS:    95%+ of values are unique within the file
  3. FORMAT:        values look like structured codes (alphanumeric patterns)

A column passing 2+ signals is high-confidence. User confirms before any merge.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import pandas as pd


class CrossFileDeduplicator:
    """Detect identifier columns and find rows sharing IDs across files."""

    # Substrings that suggest a column holds an identifier.
    # Broad on purpose — clients across industries use different vocabulary.
    NAME_TOKENS = (
        "id", "no", "number", "num", "code", "ref", "reference", "key",
        "sku", "uuid", "guid",
        "invoice", "bill", "receipt",
        "account", "acct",
        "order", "transaction", "txn", "contract",
        "passport", "license", "licence", "plate", "vin", "imei",
        "emirates_id", "national_id", "tax_id", "vat",
    )

    # Tokens that should DISQUALIFY a column even if it contains an ID-ish name.
    # These are obviously not identifiers (timestamps, flags, descriptive text).
    DISQUALIFIERS = (
        "_at", "_time", "_date", "_on",
        "name", "title", "description", "notes",
        "address", "city", "country", "region",
        "amount", "total", "price", "cost", "balance",
        "status", "type", "category",
        "valid", "verified", "opt_out", "active", "enabled",
    )

    UNIQUENESS_FOR_ID = 0.95

    # ─── Public API ───────────────────────────────────────────────────────

    def detect_id_columns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Find identifier-like columns in a single DataFrame.

        Returns a list of dicts, sorted by confidence:
          [
            {"column": "Invoice_Reference", "signals": ["name", "uniqueness", "format"],
             "confidence": "high", "uniqueness": 1.0, "sample_values": [...]},
            ...
          ]
        """
        if df.empty:
            return []

        n_rows = len(df)
        candidates: list[dict[str, Any]] = []

        for col in df.columns:
            col_str = str(col)
            lower = col_str.lower()

            # Hard disqualifier check first
            if any(d in lower for d in self.DISQUALIFIERS):
                # Some disqualifiers are too broad (e.g. "name" inside "user_name_id").
                # If the name ALSO ends with a strong ID token, allow it.
                ends_with_id = lower.endswith(("_id", "_no", "_code", "_ref", "_key", "_number"))
                if not ends_with_id:
                    continue

            signals: list[str] = []

            # Signal 1 — name
            tokens = re.split(r"[_\-\s]+", lower)
            name_match = any(t in self.NAME_TOKENS for t in tokens) or any(
                tk in lower for tk in ("invoice", "bill", "receipt", "passport", "plate", "sku", "vin")
            )
            if name_match:
                signals.append("name")

            # Signal 2 — uniqueness (skip if column is constant or mostly null)
            non_null = df[col].dropna()
            if len(non_null) >= max(2, n_rows * 0.5):
                uniqueness = non_null.nunique() / len(non_null)
                if uniqueness >= self.UNIQUENESS_FOR_ID:
                    signals.append("uniqueness")
            else:
                uniqueness = 0.0

            # Signal 3 — format (structured codes: alphanumeric, often with separators)
            if len(non_null) >= 5:
                sample = non_null.astype(str).head(50)
                format_hits = sum(
                    1 for v in sample
                    if self._looks_like_id_format(v)
                )
                if format_hits / len(sample) >= 0.7:
                    signals.append("format")

            if not signals:
                continue

            # An ID column must show real ID-like evidence, not just uniqueness.
            # Uniqueness alone is too easy — every row of a list of unique names
            # passes it. Real IDs reveal themselves through EITHER:
            #   (a) name signal + at least one other signal, OR
            #   (b) format signal alone (structured codes like INV-2024-001
            #       — plain person names cannot fake this)
            has_name = "name" in signals
            has_format = "format" in signals
            has_uniqueness = "uniqueness" in signals

            qualifies = (
                (has_name and (has_uniqueness or has_format))
                or has_format
            )
            if not qualifies:
                continue

            # High confidence: format + at least one other, OR all three
            confidence = "high" if (has_format and len(signals) >= 2) or len(signals) == 3 else "medium"

            candidates.append({
                "column": col_str,
                "signals": signals,
                "confidence": confidence,
                "uniqueness": round(uniqueness, 3),
                "sample_values": non_null.astype(str).head(3).tolist(),
            })

        # Sort: high confidence first, then by uniqueness
        candidates.sort(key=lambda c: (-len(c["signals"]), -c["uniqueness"]))
        return candidates

    def find_duplicates(
        self,
        datasets: list[dict[str, Any]],
        id_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find rows that share an ID across multiple datasets.

        Args:
            datasets: list of dicts:
              [{"dataset_id": str, "name": str, "dataframe": pd.DataFrame}, ...]
            id_columns: which columns to use for matching. If None, auto-detect
              columns common to 2+ files.

        Returns the full match analysis (see module docstring).
        """
        if len(datasets) < 2:
            return self._empty_result("Need at least 2 datasets to find cross-file duplicates.")

        # Detect ID candidates per dataset (always done, for the UI)
        dataset_info = []
        for d in datasets:
            df = d["dataframe"]
            detected = self.detect_id_columns(df)
            dataset_info.append({
                "dataset_id": d["dataset_id"],
                "name": d.get("name", d["dataset_id"]),
                "rows": int(len(df)),
                "detected_id_columns": detected,
                "dataframe": df,
            })

        # If caller didn't pick columns, use those shared across 2+ files (by lowercase name)
        if id_columns is None:
            shared = self._find_shared_id_columns(dataset_info)
            if not shared:
                return self._empty_result(
                    "No shared identifier columns auto-detected across the datasets. "
                    "Per-dataset detected IDs: "
                    + "; ".join(
                        f"{d['name']}: " + ", ".join(c["column"] for c in d["detected_id_columns"])
                        for d in dataset_info
                    )
                )
            chosen_columns = shared  # dict canonical_lower → {dataset_id: real_col}
        else:
            # User explicitly picked columns — map them to per-dataset real names (case-insensitive)
            chosen_columns = self._resolve_user_chosen_columns(id_columns, dataset_info)

        # Build clusters
        all_clusters: list[dict[str, Any]] = []
        for canonical, per_dataset_col in chosen_columns.items():
            clusters = self._build_clusters(dataset_info, canonical, per_dataset_col)
            all_clusters.extend(clusters)

        for cluster in all_clusters:
            cluster["recommended_keep"] = self._pick_best_row(cluster["occurrences"])

        total_dup_rows = sum(len(c["occurrences"]) for c in all_clusters)
        archived = max(0, total_dup_rows - len(all_clusters))

        return {
            "datasets": [
                {k: v for k, v in d.items() if k != "dataframe"}
                for d in dataset_info
            ],
            "id_columns_used": list(chosen_columns.keys()),
            "duplicate_clusters": all_clusters[:200],
            "total_clusters": len(all_clusters),
            "summary": {
                "total_clusters": len(all_clusters),
                "total_duplicate_rows": total_dup_rows,
                "rows_that_would_be_archived": archived,
            },
        }

    # ─── Format detection helper ──────────────────────────────────────────

    @staticmethod
    def _looks_like_id_format(value: str) -> bool:
        """True if value looks like a structured identifier code.

        Examples that match: INV-2024-001, BILL_8847, C-042, SKU123, 4912-AB
        Examples that don't: "Hello world", "+971501234567" (phone), "ahmed@x.com"
        """
        v = value.strip()
        if not v or len(v) > 50:
            return False
        # Strip common separators and check what's left
        compact = re.sub(r"[\s\-_/.:]", "", v)
        if len(compact) < 3:
            return False
        has_letter = bool(re.search(r"[A-Za-z]", compact))
        has_digit = bool(re.search(r"\d", compact))
        all_alnum = bool(re.fullmatch(r"[A-Za-z0-9]+", compact))
        if not all_alnum:
            return False
        # Phones, emails, and free text would have been rejected by all_alnum.
        # Structured IDs typically mix letters+digits OR are pure numeric with
        # consistent length. Pure-letter strings are usually not IDs.
        return has_digit and (has_letter or 4 <= len(compact) <= 20)

    # ─── Column matching across datasets ──────────────────────────────────

    @staticmethod
    def _find_shared_id_columns(
        dataset_info: list[dict[str, Any]],
    ) -> dict[str, dict[str, str]]:
        """Find detected ID columns whose lowercase name appears in 2+ datasets."""
        by_canonical: dict[str, dict[str, str]] = defaultdict(dict)
        for d in dataset_info:
            for cand in d["detected_id_columns"]:
                canonical = cand["column"].lower().strip()
                by_canonical[canonical][d["dataset_id"]] = cand["column"]
        return {c: m for c, m in by_canonical.items() if len(m) >= 2}

    @staticmethod
    def _resolve_user_chosen_columns(
        chosen: list[str],
        dataset_info: list[dict[str, Any]],
    ) -> dict[str, dict[str, str]]:
        """When the user explicitly picks columns, resolve them per dataset (case-insensitive)."""
        result: dict[str, dict[str, str]] = {}
        for choice in chosen:
            lower = choice.lower().strip()
            per_dataset: dict[str, str] = {}
            for d in dataset_info:
                # Find a real column name in this dataset matching the choice
                lower_to_real = {str(c).lower(): str(c) for c in d["dataframe"].columns}
                if lower in lower_to_real:
                    per_dataset[d["dataset_id"]] = lower_to_real[lower]
            if len(per_dataset) >= 2:
                result[lower] = per_dataset
        return result

    # ─── Cluster construction + best-row selection ────────────────────────

    def _build_clusters(
        self,
        dataset_info: list[dict[str, Any]],
        canonical_col: str,
        per_dataset_col: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Group rows by ID value across the given datasets."""
        # value → list of occurrences
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for d in dataset_info:
            ds_id = d["dataset_id"]
            if ds_id not in per_dataset_col:
                continue
            real_col = per_dataset_col[ds_id]
            df = d["dataframe"]

            for row_idx, value in df[real_col].items():
                if pd.isna(value) or str(value).strip() == "":
                    continue
                key = str(value).strip().lower()
                row_data = df.iloc[row_idx].fillna("").astype(str).to_dict()
                completeness = sum(1 for v in row_data.values() if v.strip()) / len(row_data) if row_data else 0
                groups[key].append({
                    "dataset_id": ds_id,
                    "dataset_name": d["name"],
                    "row_index": int(row_idx),
                    "id_value": str(value),
                    "completeness_score": round(completeness, 3),
                    "row_data": row_data,
                })

        # Keep only groups that have rows from MULTIPLE datasets (true cross-file duplicates)
        clusters: list[dict[str, Any]] = []
        for id_value, occurrences in groups.items():
            datasets_seen = {o["dataset_id"] for o in occurrences}
            if len(datasets_seen) < 2:
                continue
            clusters.append({
                "id_column": canonical_col,
                "id_value": occurrences[0]["id_value"],
                "occurrences": occurrences,
            })
        return clusters

    @staticmethod
    def _pick_best_row(occurrences: list[dict[str, Any]]) -> dict[str, Any]:
        """Pick which copy to keep. Most complete row wins; ties broken by first dataset."""
        best = max(occurrences, key=lambda o: (o["completeness_score"], -occurrences.index(o)))
        return {"dataset_id": best["dataset_id"], "row_index": best["row_index"]}

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result(reason: str) -> dict[str, Any]:
        return {
            "datasets": [],
            "id_columns_used": [],
            "duplicate_clusters": [],
            "total_clusters": 0,
            "summary": {
                "total_clusters": 0,
                "total_duplicate_rows": 0,
                "rows_that_would_be_archived": 0,
            },
            "reason": reason,
        }
