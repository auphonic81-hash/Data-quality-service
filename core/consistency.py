"""Cross-field record consistency validator.

Detects when fields within the same row don't belong together.
For example: name='Ahmed' but email='sara@gmail.com' suggests data corruption.

This is a HIGH-VALUE enterprise feature (Informatica calls it cross-field validation).
We DO NOT auto-fix these issues — we surface them for human review, because
"fixing" them automatically could destroy legitimate aliases or business cases.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from rapidfuzz import fuzz


class RecordConsistencyChecker:
    """Validates that fields within a row are mutually consistent."""

    # Minimum similarity score to consider an email-name pair "matching"
    EMAIL_NAME_THRESHOLD = 60

    def check(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all consistency checks on a DataFrame."""
        if df.empty:
            return {"checked_rows": 0, "issues": []}

        name_col = self._find_column(df, ["name", "full_name", "fullname", "first_name"])
        email_col = self._find_column(df, ["email", "email_address"])

        issues: list[dict[str, Any]] = []

        if name_col and email_col:
            issues.extend(self._check_name_email_consistency(df, name_col, email_col))

        return {
            "checked_rows": len(df),
            "checks_performed": [
                f"name_email_match ({name_col} ↔ {email_col})" if name_col and email_col else None,
            ],
            "issue_count": len(issues),
            "issues": issues[:100],  # cap to keep payload small
        }

    def _check_name_email_consistency(
        self, df: pd.DataFrame, name_col: str, email_col: str
    ) -> list[dict[str, Any]]:
        """Flag rows where the email local-part doesn't relate to the name."""
        issues: list[dict[str, Any]] = []

        for idx, row in df.iterrows():
            name = row.get(name_col)
            email = row.get(email_col)
            if pd.isna(name) or pd.isna(email):
                continue

            name_str = str(name).strip().lower()
            email_str = str(email).strip().lower()

            # Extract local part (before @)
            if "@" not in email_str:
                continue
            local_part = email_str.split("@", 1)[0]

            # Normalize: drop dots, digits, underscores, hyphens
            local_tokens = re.split(r"[._\-+0-9]+", local_part)
            local_tokens = [t for t in local_tokens if t]

            # Name tokens
            name_tokens = [t for t in re.split(r"\s+", name_str) if len(t) >= 2]

            if not local_tokens or not name_tokens:
                continue

            # For each local-part token, find best match against name tokens
            best_score = 0
            for lt in local_tokens:
                for nt in name_tokens:
                    score = max(
                        fuzz.ratio(lt, nt),
                        fuzz.partial_ratio(lt, nt),
                    )
                    if score > best_score:
                        best_score = score

            if best_score < self.EMAIL_NAME_THRESHOLD:
                issues.append({
                    "row_index": int(idx),
                    "type": "name_email_mismatch",
                    "name": str(name),
                    "email": str(email),
                    "match_score": int(best_score),
                    "severity": "high" if best_score < 30 else "medium",
                })

        return issues

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """Find a column by case-insensitive name match."""
        lower_map = {c.lower(): c for c in df.columns}
        for candidate in candidates:
            if candidate in lower_map:
                return lower_map[candidate]
        # Partial match fallback
        for candidate in candidates:
            for col_lower, col in lower_map.items():
                if candidate in col_lower:
                    return col
        return None
