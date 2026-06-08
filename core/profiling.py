"""Data profiling module.

Wraps ydata-profiling with custom analyzers for:
- Phone number format detection (Gulf-aware)
- Email validity per record
- Arabic text detection
- Date format consistency
- Hidden duplicates
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from ydata_profiling import ProfileReport


class DataProfiler:
    """Generates comprehensive data profiles using ydata-profiling + custom checks."""

    # Regex patterns for format detection
    PHONE_PATTERNS = {
        "uae_local": r"^\+?9715\d{8}$",
        "uae_landline": r"^\+?9714\d{7}$",
        "saudi": r"^\+?9665\d{8}$",
        "gulf_generic": r"^\+?9\d{10,12}$",
        "international": r"^\+\d{10,15}$",
    }
    EMAIL_PATTERN = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    ARABIC_PATTERN = r"[\u0600-\u06FF]"
    DATE_PATTERNS = [
        r"^\d{4}-\d{2}-\d{2}",                 # YYYY-MM-DD
        r"^\d{2}/\d{2}/\d{4}",                 # DD/MM/YYYY or MM/DD/YYYY
        r"^\d{2}-\d{2}-\d{4}",                 # DD-MM-YYYY
        r"^\d{4}/\d{2}/\d{2}",                 # YYYY/MM/DD
    ]

    def __init__(self, sample_size: int = 10000, minimal: bool = True):
        self.sample_size = sample_size
        self.minimal = minimal

    def profile(self, df: pd.DataFrame, title: str = "Data Quality Report") -> dict[str, Any]:
        """Run full profiling: ydata-profiling + custom checks."""
        if df.empty:
            raise ValueError("Cannot profile an empty DataFrame")

        sampled = self._sample(df)

        report = ProfileReport(sampled, title=title, minimal=self.minimal)
        ydata_summary = self._extract_ydata_summary(report)
        custom_findings = self._run_custom_checks(sampled)

        return {
            "title": title,
            "row_count": len(df),
            "column_count": len(df.columns),
            "sampled_rows": len(sampled),
            "ydata_summary": ydata_summary,
            "custom_findings": custom_findings,
            "_report": report,
        }

    def save_report(self, profile_result: dict[str, Any], output_path: str | Path) -> Path:
        """Save the HTML report to disk."""
        report = profile_result.get("_report")
        if report is None:
            raise ValueError("No report object available in profile result")
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_file(path)
        return path

    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample the DataFrame if it exceeds sample_size."""
        if len(df) <= self.sample_size:
            return df
        return df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)

    def _extract_ydata_summary(self, report: ProfileReport) -> dict[str, Any]:
        """Extract key metrics from ydata report."""
        try:
            description = report.description_set
            table = description.table if hasattr(description, "table") else {}
            alerts = description.alerts if hasattr(description, "alerts") else []
            return {
                "n_rows": table.get("n", 0),
                "n_columns": table.get("n_var", 0),
                "missing_cells": table.get("n_cells_missing", 0),
                "missing_pct": round(table.get("p_cells_missing", 0) * 100, 2),
                "duplicate_rows": table.get("n_duplicates", 0),
                "duplicate_pct": round(table.get("p_duplicates", 0) * 100, 2),
                "alerts_count": len(alerts),
                "alerts": [str(a) for a in alerts[:50]],
            }
        except Exception as exc:
            return {"error": f"Could not extract ydata summary: {exc}"}

    def _run_custom_checks(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run custom domain-specific checks on each column."""
        findings: dict[str, dict[str, Any]] = {}

        for column in df.columns:
            series = df[column].dropna().astype(str)
            if series.empty:
                continue

            column_findings: dict[str, Any] = {}

            # Detect column intent — order matters: name hints first, then dates,
            # then email, then phone (phones are the most permissive regex).
            col_lower = column.lower()
            if any(t in col_lower for t in ("date", "time", "_at", "birthday", "dob")):
                if self._looks_like_date(series):
                    column_findings["detected_type"] = "date"
                    column_findings["format_consistency"] = self._check_date_formats(series)
            elif any(t in col_lower for t in ("email", "mail")):
                column_findings["detected_type"] = "email"
                column_findings["valid_emails_pct"] = self._check_email_validity(series)
            elif any(t in col_lower for t in ("phone", "mobile", "fax", "tel")):
                column_findings["detected_type"] = "phone"
                column_findings["format_consistency"] = self._check_phone_formats(series)
            elif self._looks_like_date(series):
                column_findings["detected_type"] = "date"
                column_findings["format_consistency"] = self._check_date_formats(series)
            elif self._looks_like_email(series):
                column_findings["detected_type"] = "email"
                column_findings["valid_emails_pct"] = self._check_email_validity(series)
            elif self._looks_like_phone(series):
                column_findings["detected_type"] = "phone"
                column_findings["format_consistency"] = self._check_phone_formats(series)

            # Arabic content detection
            arabic_pct = self._arabic_content_pct(series)
            if arabic_pct > 0:
                column_findings["arabic_content_pct"] = arabic_pct

            # Suspicious patterns
            if self._has_placeholder_values(series):
                column_findings["has_placeholder_values"] = True

            if column_findings:
                findings[column] = column_findings

        return findings

    @staticmethod
    def _looks_like_phone(series: pd.Series) -> bool:
        sample = series.head(50)
        phone_like = sum(
            1 for v in sample
            if re.match(r"^[\+\d\s\-\(\)]{7,20}$", str(v).strip())
        )
        return phone_like / max(len(sample), 1) > 0.7

    @staticmethod
    def _looks_like_email(series: pd.Series) -> bool:
        sample = series.head(50)
        email_like = sum(1 for v in sample if "@" in str(v) and "." in str(v))
        return email_like / max(len(sample), 1) > 0.7

    @staticmethod
    def _looks_like_date(series: pd.Series) -> bool:
        sample = series.head(50).astype(str)
        date_like = sum(
            1 for v in sample
            if any(re.match(p, v.strip()) for p in DataProfiler.DATE_PATTERNS)
        )
        return date_like / max(len(sample), 1) > 0.7

    def _check_phone_formats(self, series: pd.Series) -> dict[str, Any]:
        """Check phone number format consistency."""
        normalized = series.str.replace(r"[\s\-\(\)]", "", regex=True)
        format_counts: dict[str, int] = {"unknown": 0}
        for name in self.PHONE_PATTERNS:
            format_counts[name] = 0

        for value in normalized:
            matched = False
            for name, pattern in self.PHONE_PATTERNS.items():
                if re.match(pattern, value):
                    format_counts[name] += 1
                    matched = True
                    break
            if not matched:
                format_counts["unknown"] += 1

        total = len(normalized)
        return {
            "total": total,
            "by_format": format_counts,
            "consistency_pct": round(
                (total - format_counts["unknown"]) / total * 100, 2
            ) if total else 0,
        }

    def _check_email_validity(self, series: pd.Series) -> float:
        """Return percentage of valid emails."""
        valid = sum(1 for v in series if re.match(self.EMAIL_PATTERN, str(v).strip()))
        return round(valid / len(series) * 100, 2)

    def _check_date_formats(self, series: pd.Series) -> dict[str, Any]:
        """Check date format consistency."""
        format_counts: dict[str, int] = {pattern: 0 for pattern in self.DATE_PATTERNS}
        format_counts["unknown"] = 0

        for value in series:
            matched = False
            for pattern in self.DATE_PATTERNS:
                if re.match(pattern, str(value).strip()):
                    format_counts[pattern] += 1
                    matched = True
                    break
            if not matched:
                format_counts["unknown"] += 1

        total = len(series)
        consistency = (total - format_counts["unknown"]) / total * 100 if total else 0
        return {
            "total": total,
            "by_format": format_counts,
            "consistency_pct": round(consistency, 2),
        }

    def _arabic_content_pct(self, series: pd.Series) -> float:
        """Percentage of values containing Arabic characters."""
        arabic = sum(1 for v in series if re.search(self.ARABIC_PATTERN, str(v)))
        return round(arabic / len(series) * 100, 2)

    @staticmethod
    def _has_placeholder_values(series: pd.Series) -> bool:
        """Detect placeholder/sample values."""
        placeholders = {"sample", "test", "n/a", "null", "none", "tbd", "xxx", "yyy"}
        sample_lower = series.head(20).str.lower()
        for value in sample_lower:
            if any(p in value for p in placeholders):
                return True
        return False