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



COUNTRY_TO_ISO: dict[str, str] = {
    "USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US", "US": "US",
    "UK": "GB", "UNITED KINGDOM": "GB", "GREAT BRITAIN": "GB", "ENGLAND": "GB",
    "UAE": "AE", "UNITED ARAB EMIRATES": "AE", "EMIRATES": "AE",
    "SAUDI ARABIA": "SA", "SAUDI": "SA", "KSA": "SA",
    "FRANCE": "FR", "GERMANY": "DE", "ITALY": "IT", "SPAIN": "ES",
    "NETHERLANDS": "NL", "BELGIUM": "BE", "SWEDEN": "SE", "NORWAY": "NO",
    "DENMARK": "DK", "FINLAND": "FI", "POLAND": "PL", "PORTUGAL": "PT",
    "AUSTRIA": "AT", "SWITZERLAND": "CH", "IRELAND": "IE",
    "CANADA": "CA", "MEXICO": "MX", "BRAZIL": "BR", "ARGENTINA": "AR",
    "AUSTRALIA": "AU", "NEW ZEALAND": "NZ",
    "CHINA": "CN", "JAPAN": "JP", "SOUTH KOREA": "KR", "KOREA": "KR",
    "INDIA": "IN", "PAKISTAN": "PK", "BANGLADESH": "BD",
    "SINGAPORE": "SG", "PHILIPPINES": "PH", "INDONESIA": "ID", "THAILAND": "TH",
    "VIETNAM": "VN", "MALAYSIA": "MY", "HONG KONG": "HK", "TAIWAN": "TW",
    "EGYPT": "EG", "JORDAN": "JO", "LEBANON": "LB", "TURKEY": "TR",
    "ISRAEL": "IL", "QATAR": "QA", "KUWAIT": "KW", "BAHRAIN": "BH", "OMAN": "OM",
    "SOUTH AFRICA": "ZA", "NIGERIA": "NG", "KENYA": "KE", "MOROCCO": "MA",
    "RUSSIA": "RU", "UKRAINE": "UA",
}


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
        # Pre-pass: strip thousands-separator commas from numeric strings
        # so downstream coercion (pandera) sees clean numbers
        import re as _re
        _num_with_comma = _re.compile(r"^\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*$")
        for _col in cleaned.columns:
            if cleaned[_col].dtype == object:
                cleaned[_col] = cleaned[_col].apply(
                    lambda v: v.replace(",", "") if isinstance(v, str) and _num_with_comma.match(v) else v
                )
        change_log: dict[str, dict[str, Any]] = {}

        for col in cleaned.columns:
            col_type = (column_types or {}).get(col) or self._auto_detect_type(cleaned[col])

            if col_type == "phone":
                cleaned[col], log = self._standardize_phones(cleaned[col], cleaned)
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
            elif col_type == "status":
                cleaned[col], log = self._standardize_status(cleaned[col])
                if log["changes"] > 0:
                    change_log[col] = {"type": "status", **log}
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

    def _standardize_phones(self, series, full_df=None):
        """Standardize phones using phonenumbers library (Google\'s).

        Tries the row\'s Country column (if present), then a fallback list.
        Format E.164. Any number that can\'t be parsed becomes a failure.
        """
        import phonenumbers
        from phonenumbers import NumberParseException

        COUNTRY_ALIASES = {
            "uae":"AE","united arab emirates":"AE","u.a.e.":"AE","ae":"AE",
            "saudi arabia":"SA","ksa":"SA","saudi":"SA","sa":"SA",
            "germany":"DE","de":"DE","france":"FR","fr":"FR",
            "usa":"US","united states":"US","us":"US",
            "uk":"GB","united kingdom":"GB","gb":"GB","britain":"GB",
            "japan":"JP","jp":"JP","spain":"ES","es":"ES",
        }
        TRY_ORDER = ["AE","SA","US","GB","DE","FR","JP","ES"]

        def resolve(raw):
            if not raw or not isinstance(raw, str): return None
            return COUNTRY_ALIASES.get(raw.strip().lower())

        # Find a country column in full_df
        country_col = None
        if full_df is not None:
            for c in full_df.columns:
                if c.lower() in ("country","country_code","country_name","nation"):
                    country_col = c
                    break

        out = series.astype(object).copy()
        changes = 0
        failures = 0
        for idx, val in series.items():
            if val is None or (isinstance(val,float) and pd.isna(val)):
                continue
            s = str(val).strip()
            if not s:
                continue
            candidates = []
            if country_col is not None:
                c = resolve(full_df.at[idx, country_col])
                if c: candidates.append(c)
            for c in TRY_ORDER:
                if c not in candidates:
                    candidates.append(c)
            parsed = None
            # If the value already starts with +, parse WITHOUT a region hint
            # (the + tells phonenumbers it\'s already international format).
            # Adding a region hint can cause double-prefix or wrong parsing.
            if s.startswith("+"):
                try:
                    p = phonenumbers.parse(s, None)
                    if phonenumbers.is_valid_number(p):
                        parsed = p
                except NumberParseException:
                    pass
            # Otherwise, try region candidates
            if parsed is None:
                for c in candidates:
                    try:
                        p = phonenumbers.parse(s, c)
                        if phonenumbers.is_valid_number(p):
                            parsed = p
                            break
                    except NumberParseException:
                        continue
            # Final fallback: try None region
            if parsed is None:
                try:
                    p = phonenumbers.parse(s, None)
                    if phonenumbers.is_valid_number(p):
                        parsed = p
                except NumberParseException:
                    pass
            if parsed is not None:
                formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                if formatted != s:
                    out.at[idx] = formatted
                    changes += 1
            else:
                failures += 1
        return out, {"changes": changes, "failures": failures}


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

    def _standardize_names(self, series, full_df=None):
        """Title-case names, preserving common all-caps acronyms (NY, IT, CEO)."""
        ACRONYMS = {"NY","LA","CA","UK","US","USA","UAE","EU","IT","HR","CEO","CFO","CTO","COO","VIP","R&D"}
        def fix(v):
            if v is None or (isinstance(v, float) and pd.isna(v)): return v
            s = str(v).strip()
            if not s: return v
            if not (s.isupper() or s.islower()): return s
            parts = []
            for w in s.split():
                if w.upper() in ACRONYMS:
                    parts.append(w.upper())
                else:
                    parts.append(w.capitalize())
            return " ".join(parts)
        out = series.astype(object).copy()
        changes = 0
        for idx, v in series.items():
            fixed = fix(v)
            if fixed != v and not (isinstance(v,float) and pd.isna(v)):
                out.at[idx] = fixed
                changes += 1
        return out, {"changes": changes, "failures": 0}


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

    STATUS_CODE_MAP = {
        # Employee
        "ACT": "Active", "INA": "Inactive", "TRM": "Terminated",
        "LOA": "On Leave", "PND": "Pending",
        # Invoice
        "PAID": "Paid", "UNPAID": "Unpaid", "OVD": "Overdue",
        "DUE": "Pending", "CNL": "Cancelled", "CANCELLED": "Cancelled",
        # Generic short forms
        "A": "Active", "I": "Inactive", "P": "Paid", "U": "Unpaid",
        "Y": "Active", "N": "Inactive",
    }

    def _standardize_status(self, series):
        """Translate status codes to readable labels and enforce Title case.
        
        Two-pass normalization:
        1. Look up known short codes in STATUS_CODE_MAP (ACT->Active, PAID->Paid)
        2. Title-case ALL values so 'active'/'ACTIVE'/'Active' all become 'Active'
        
        Semantic translation (e.g. invoice Active->Open) happens later in
        master_repository, where entity context is known.
        """
        out = series.astype(object).copy()
        changes = 0
        for idx, val in series.items():
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            s = str(val).strip()
            if not s:
                out.at[idx] = "Unknown"
                changes += 1
                continue
            # Pass 1: short code translation
            mapped = self.STATUS_CODE_MAP.get(s.upper(), s)
            # Pass 2: enforce Title case for any multi-word or single-word value
            # "On Leave" stays "On Leave", "active" -> "Active", "ACTIVE" -> "Active"
            normalized = " ".join(w.capitalize() for w in mapped.split())
            if normalized != s:
                out.at[idx] = normalized
                changes += 1
        return out, {"changes": changes}

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
        if any(t in name for t in ("status", "state", "status_code", "status_cd")):
            return "status"

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