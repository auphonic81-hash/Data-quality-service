"""Data remediation module.

Automatically fixes detected data quality issues:
- Phone number standardization (Gulf-aware)
- Date format normalization
- Email cleaning
- Name standardization (Arabic + English)
- Whitespace and case normalization
- Duplicate clustering and merging
"""
from __future__ import annotations

import re
from typing import Any

import dateparser
import pandas as pd
import phonenumbers
from email_validator import EmailNotValidError, validate_email
from rapidfuzz import fuzz, process


class DataRemediator:
    """Applies automated fixes to detected data quality issues."""

    DEFAULT_REGION = "AE"  # UAE
    CLUSTER_THRESHOLD = 75

    def __init__(self, default_region: str = "AE"):
        self.default_region = default_region

    def remediate(
        self,
        df: pd.DataFrame,
        column_types: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Apply remediation across all relevant columns.

        Args:
            df: Source DataFrame.
            column_types: Optional override mapping column name → detected type
                          (e.g. {"Phone": "phone", "Email": "email", "DOB": "date"}).

        Returns:
            (cleaned_dataframe, change_log)
        """
        cleaned = df.copy()
        change_log: dict[str, dict[str, Any]] = {}

        for col in cleaned.columns:
            col_type = (column_types or {}).get(col) or self._auto_detect_type(cleaned[col])

            if col_type == "phone":
                cleaned[col], log = self._standardize_phones(cleaned[col])
                change_log[col] = {"type": "phone", **log}
            elif col_type == "email":
                cleaned[col], log = self._standardize_emails(cleaned[col])
                change_log[col] = {"type": "email", **log}
            elif col_type == "date":
                cleaned[col], log = self._standardize_dates(cleaned[col])
                change_log[col] = {"type": "date", **log}
            elif col_type == "name":
                cleaned[col], log = self._standardize_names(cleaned[col])
                change_log[col] = {"type": "name", **log}
            elif col_type == "text":
                cleaned[col], log = self._clean_text(cleaned[col])
                if log["changes"] > 0:
                    change_log[col] = {"type": "text", **log}

        return cleaned, change_log

    def cluster_similar_values(
        self, series: pd.Series, threshold: int | None = None
    ) -> dict[str, list[str]]:
        """Cluster similar values (e.g. 'Ahmed', 'Amed', 'Ahmd' → one cluster)."""
        threshold = threshold or self.CLUSTER_THRESHOLD
        unique_values = series.dropna().astype(str).unique().tolist()
        clusters: dict[str, list[str]] = {}
        assigned: set[str] = set()

        for value in unique_values:
            if value in assigned:
                continue
            matches = process.extract(value, unique_values, scorer=fuzz.ratio, limit=10)
            similar = [match for match, score, _ in matches if score >= threshold and match != value]
            if similar:
                clusters[value] = similar
                assigned.update(similar)
                assigned.add(value)

        return clusters

    # ─── Type-specific cleaners ────────────────────────────────────────────

    def _standardize_phones(self, series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        """Normalize phone numbers to E.164 format."""
        before = series.copy()
        cleaned: list[str | None] = []
        failures = 0

        for value in series:
            if pd.isna(value):
                cleaned.append(None)
                continue
            try:
                parsed = phonenumbers.parse(str(value), self.default_region)
                if phonenumbers.is_valid_number(parsed):
                    formatted = phonenumbers.format_number(
                        parsed, phonenumbers.PhoneNumberFormat.E164
                    )
                    cleaned.append(formatted)
                else:
                    cleaned.append(str(value))
                    failures += 1
            except phonenumbers.NumberParseException:
                cleaned.append(str(value))
                failures += 1

        new_series = pd.Series(cleaned, index=series.index)
        changed = (before.fillna("").astype(str) != new_series.fillna("").astype(str)).sum()

        return new_series, {
            "changes": int(changed),
            "failures": failures,
        }

    def _standardize_emails(self, series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        """Normalize and validate emails."""
        before = series.copy()
        cleaned: list[str | None] = []
        failures = 0

        for value in series:
            if pd.isna(value):
                cleaned.append(None)
                continue
            try:
                # check_deliverability=False because we may be offline / in sandbox
                result = validate_email(str(value).strip().lower(), check_deliverability=False)
                cleaned.append(result.normalized)
            except EmailNotValidError:
                cleaned.append(str(value))
                failures += 1

        new_series = pd.Series(cleaned, index=series.index)
        changed = (before.fillna("").astype(str) != new_series.fillna("").astype(str)).sum()

        return new_series, {
            "changes": int(changed),
            "failures": failures,
        }

    def _standardize_dates(self, series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        """Parse and normalize dates to ISO format."""
        before = series.copy()
        cleaned: list[str | None] = []
        failures = 0

        for value in series:
            if pd.isna(value):
                cleaned.append(None)
                continue
            try:
                parsed = dateparser.parse(str(value))
                if parsed:
                    cleaned.append(parsed.strftime("%Y-%m-%d"))
                else:
                    cleaned.append(str(value))
                    failures += 1
            except Exception:
                cleaned.append(str(value))
                failures += 1

        new_series = pd.Series(cleaned, index=series.index)
        changed = (before.fillna("").astype(str) != new_series.fillna("").astype(str)).sum()

        return new_series, {
            "changes": int(changed),
            "failures": failures,
        }

    def _standardize_names(self, series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        """Clean, standardize, and cluster similar names (Ahmed/Ahmd/Amed)."""
        before = series.copy()

        # Step 1: basic cleaning
        # Common acronyms / abbreviations to preserve verbatim
        ACRONYMS = {"IT", "HR", "USA", "UAE", "UK", "EU", "CEO", "CTO", "CFO",
                    "VIP", "B2B", "B2C", "SQL", "API", "AI", "ML", "PR"}

        def clean_one(value: Any) -> Any:
            if pd.isna(value):
                return value
            text = str(value).strip()
            text = re.sub(r"\s+", " ", text)
            if re.search(r"[\u0600-\u06FF]", text):
                return text
            # Title-case each token but preserve acronyms (2-4 letter all-caps)
            tokens = text.split()
            out = []
            for tok in tokens:
                stripped = re.sub(r"[^A-Za-z]", "", tok)
                if stripped.upper() in ACRONYMS or (
                    len(stripped) <= 4 and stripped.isalpha() and stripped.isupper()
                ):
                    out.append(tok.upper())
                else:
                    out.append(tok.capitalize())
            return " ".join(out)

        cleaned = series.apply(clean_one)

        # Step 2: smart fuzzy clustering with token-level logic
        # Only merge names where structure matches (same last name, similar first name)
        non_null = cleaned.dropna().astype(str)
        unique_values = non_null.value_counts()
        value_list = unique_values.index.tolist()
        replacement_map: dict[str, str] = {}

        def name_parts(name: str) -> tuple[str, str]:
            """Split into (first_token, rest). Returns lowercase for comparison."""
            tokens = name.lower().split()
            if not tokens:
                return ("", "")
            if len(tokens) == 1:
                return (tokens[0], "")
            return (tokens[0], " ".join(tokens[1:]))

        def safe_to_merge(a: str, b: str) -> bool:
            """Only merge if ALL tokens have high similarity individually."""
            a_tokens = a.lower().split()
            b_tokens = b.lower().split()
            if len(a_tokens) != len(b_tokens):
                return False
            # Each corresponding token must be >= 82% similar
            for ta, tb in zip(a_tokens, b_tokens):
                if fuzz.ratio(ta, tb) < 82:
                    return False
            # And the overall string must still be >= 80% similar
            return fuzz.ratio(a.lower(), b.lower()) >= 80

        for value in value_list:
            if value in replacement_map:
                continue
            candidates = process.extract(value, value_list, scorer=fuzz.ratio, limit=20)
            cluster = [value]
            for candidate, _, _ in candidates:
                if candidate == value or candidate in replacement_map:
                    continue
                if safe_to_merge(value, candidate):
                    cluster.append(candidate)
            if len(cluster) > 1:
                # Pick the longest variant (most complete spelling)
                canonical = max(cluster, key=lambda v: (len(v), unique_values.get(v, 0)))
                for variant in cluster:
                    if variant != canonical:
                        replacement_map[variant] = canonical

        clusters_applied = len(replacement_map)
        new_series = cleaned.replace(replacement_map)

        changed = (before.fillna("").astype(str) != new_series.fillna("").astype(str)).sum()

        return new_series, {
            "changes": int(changed),
            "failures": 0,
            "clusters_merged": clusters_applied,
        }

    def _clean_text(self, series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        """Strip whitespace, normalize internal spaces."""
        if series.dtype != object:
            return series, {"changes": 0, "failures": 0}

        before = series.copy()

        def clean_one(value: Any) -> Any:
            if pd.isna(value):
                return value
            return re.sub(r"\s+", " ", str(value).strip())

        new_series = series.apply(clean_one)
        changed = (before.fillna("").astype(str) != new_series.fillna("").astype(str)).sum()

        return new_series, {"changes": int(changed), "failures": 0}

    # ─── Auto-detection ────────────────────────────────────────────────────

    @staticmethod
    def _auto_detect_type(series: pd.Series) -> str:
        """Heuristically detect the type of a column for remediation.
        Column-name hints come first; content heuristics are a fallback."""
        if series.dtype != object:
            return "numeric"

        name = (str(series.name) if series.name is not None else "").lower()

        # Name-based hints (most reliable)
        if any(t in name for t in ("date", "time", "_at", "birthday", "dob")):
            return "date"
        if any(t in name for t in ("email", "mail")):
            return "email"
        if any(t in name for t in ("phone", "mobile", "fax", "tel")):
            return "phone"
        if any(t in name for t in ("name", "title", "owner", "contact", "person")):
            return "name"

        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return "text"

        # Date detection BEFORE phone (since dates can look numeric)
        date_patterns = [
            r"^\d{4}-\d{1,2}-\d{1,2}",
            r"^\d{1,2}/\d{1,2}/\d{4}",
            r"^\d{1,2}-\d{1,2}-\d{4}",
            r"^\d{4}/\d{1,2}/\d{1,2}",
        ]
        date_like = sum(
            1 for v in sample
            if any(re.match(p, v.strip()) for p in date_patterns)
        )
        if date_like / len(sample) > 0.7:
            return "date"

        # Email detection
        email_like = sum(1 for v in sample if "@" in v and "." in v)
        if email_like / len(sample) > 0.7:
            return "email"

        # Phone detection (strict — must start with + or 00, or be 10+ digits)
        phone_like = sum(
            1 for v in sample
            if re.match(r"^(\+|00)?[\d\s\-\(\)]{10,20}$", v.strip())
            and sum(c.isdigit() for c in v) >= 8
        )
        if phone_like / len(sample) > 0.7:
            return "phone"

        # Name detection
        name_like = sum(
            1 for v in sample
            if re.match(r"^[a-zA-Z\u0600-\u06FF\s\-\.]{2,40}$", v.strip())
        )
        if name_like / len(sample) > 0.7:
            return "name"

        return "text"