"""Quality detection module.

Identifies specific data quality issues:
- Duplicates (exact + fuzzy)
- Missing values
- Type inconsistencies
- Format violations
- Outliers
- Constraint violations
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from rapidfuzz import fuzz


class QualityDetector:
    """Detects data quality issues across multiple dimensions."""

    DUPLICATE_THRESHOLD = 85  # rapidfuzz similarity score for fuzzy matching

    def detect_all(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all quality checks and return a structured report."""
        if df.empty:
            raise ValueError("Cannot analyze an empty DataFrame")

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "exact_duplicates": self._exact_duplicates(df),
            "fuzzy_duplicates": self._fuzzy_duplicates(df),
            "missing_analysis": self._missing_analysis(df),
            "type_inconsistencies": self._type_inconsistencies(df),
            "constant_columns": self._constant_columns(df),
            "unique_columns": self._unique_columns(df),
            "outliers": self._outliers(df),
        }

    def _exact_duplicates(self, df: pd.DataFrame) -> dict[str, Any]:
        """Find exact duplicate rows."""
        duplicate_mask = df.duplicated(keep=False)
        count = int(duplicate_mask.sum())
        return {
            "count": count,
            "percentage": round(count / len(df) * 100, 2) if len(df) else 0,
            "row_indices": df[duplicate_mask].index.tolist()[:100],
        }

    def _fuzzy_duplicates(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> dict[str, Any]:
        """Find fuzzy duplicates in text columns (limited sample for performance)."""
        text_columns = columns or [
            col for col in df.columns if df[col].dtype == object
        ][:5]  # Limit to first 5 text columns for performance

        if not text_columns:
            return {"detected": 0, "samples": []}

        # Sample to keep it tractable
        sample = df[text_columns].head(500).fillna("").astype(str)
        clusters: list[dict[str, Any]] = []
        seen: set[int] = set()

        for idx, row in sample.iterrows():
            if idx in seen:
                continue
            row_text = " | ".join(row.values).lower()
            matches: list[int] = []
            for other_idx, other_row in sample.iloc[idx + 1:].iterrows():
                if other_idx in seen:
                    continue
                other_text = " | ".join(other_row.values).lower()
                score = fuzz.ratio(row_text, other_text)
                if score >= self.DUPLICATE_THRESHOLD:
                    matches.append(int(other_idx))
                    seen.add(other_idx)

            if matches:
                seen.add(idx)
                clusters.append({
                    "anchor_index": int(idx),
                    "anchor_preview": row_text[:200],
                    "similar_indices": matches,
                    "cluster_size": len(matches) + 1,
                })

            if len(clusters) >= 20:  # Cap output
                break

        return {
            "detected": len(clusters),
            "samples": clusters,
        }

    def _missing_analysis(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze missing values per column."""
        missing = df.isna().sum()
        total_cells = len(df) * len(df.columns)
        missing_total = int(missing.sum())

        per_column = {
            col: {
                "count": int(missing[col]),
                "percentage": round(missing[col] / len(df) * 100, 2),
            }
            for col in df.columns
            if missing[col] > 0
        }

        return {
            "total_missing_cells": missing_total,
            "total_missing_pct": round(missing_total / total_cells * 100, 2) if total_cells else 0,
            "columns_with_missing": len(per_column),
            "per_column": per_column,
        }

    def _type_inconsistencies(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect columns where values don't all match the expected type."""
        inconsistencies: dict[str, dict[str, Any]] = {}

        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue

            # Try to infer if column should be numeric
            if series.dtype == object:
                numeric_count = pd.to_numeric(series, errors="coerce").notna().sum()
                ratio = numeric_count / len(series)
                if 0.5 < ratio < 1.0:
                    inconsistencies[col] = {
                        "issue": "mixed_numeric_text",
                        "numeric_pct": round(ratio * 100, 2),
                        "text_pct": round((1 - ratio) * 100, 2),
                    }

        return {
            "columns_with_issues": len(inconsistencies),
            "details": inconsistencies,
        }

    def _constant_columns(self, df: pd.DataFrame) -> list[str]:
        """Find columns with a single unique value."""
        return [col for col in df.columns if df[col].nunique(dropna=False) == 1]

    def _unique_columns(self, df: pd.DataFrame) -> list[str]:
        """Find columns where every value is unique (potential ID columns)."""
        return [
            col for col in df.columns
            if df[col].nunique(dropna=False) == len(df) and len(df) > 1
        ]

    def _outliers(self, df: pd.DataFrame) -> dict[str, Any]:
        """Detect numerical outliers using IQR method."""
        outliers_per_col: dict[str, dict[str, Any]] = {}

        for col in df.select_dtypes(include=["number"]).columns:
            series = df[col].dropna()
            if len(series) < 10:
                continue

            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)
            outlier_count = int(outlier_mask.sum())

            if outlier_count > 0:
                outliers_per_col[col] = {
                    "count": outlier_count,
                    "percentage": round(outlier_count / len(series) * 100, 2),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }

        return {
            "columns_with_outliers": len(outliers_per_col),
            "details": outliers_per_col,
        }