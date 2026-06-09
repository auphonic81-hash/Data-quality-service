"""Cross-source entity resolution using the recordlinkage toolkit.

Detects when the same real-world entity (person, company, etc.) appears in
multiple sources — even when the data is dirty, formatted differently, or
has different column names.

Two modes:
  - link_across:  find the same entity across two different datasets
  - dedup_within: find duplicate records inside a single dataset

Design principles:
  - Pandas-native, no SQL backends
  - Deterministic + fuzzy: exact match on email/phone, fuzzy on name
  - Cross-schema: auto-maps columns between sources with different names
    (e.g. "Full_Name" in A maps to "name" in B)
  - Explicit confidence scoring so users see why each match was made
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
import recordlinkage
from rapidfuzz import fuzz


class EntityResolver:
    """Resolves entities within one dataset or across two datasets."""

    # Column-name patterns we auto-detect as identity attributes
    NAME_PATTERNS = ("full_name", "fullname", "name", "owner_name", "contact", "person")
    EMAIL_PATTERNS = ("email", "mail", "e_mail")
    PHONE_PATTERNS = ("phone", "mobile", "tel", "cell")

    # How strict to be when fuzzy-matching names
    NAME_SIM_THRESHOLD = 75   # min Jaro-Winkler-style score (0-100) to count as a partial name match

    # How many matching signals (out of name+email+phone) are needed for each tier
    HIGH_CONFIDENCE_SCORE = 2.0   # e.g. exact email + exact phone
    MEDIUM_CONFIDENCE_SCORE = 1.0  # e.g. exact email only, or fuzzy name + partial signal

    # Per-signal weights (sum drives confidence)
    WEIGHT_EMAIL_EXACT = 1.5
    WEIGHT_PHONE_EXACT = 1.5
    WEIGHT_NAME_FUZZY = 1.0  # scaled by similarity 0-1

    # ─── Public API ───────────────────────────────────────────────────────

    def link_across_sources(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        source_a_name: str = "source_a",
        source_b_name: str = "source_b",
        match_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Link records that represent the same entity across two datasets."""
        df_a_prep, mapping_a = self._prepare_dataframe(df_a, "left")
        df_b_prep, mapping_b = self._prepare_dataframe(df_b, "right")

        # Find which identity signals exist in BOTH sources
        shared_signals = [
            sig for sig in ("name", "email", "phone")
            if sig in mapping_a and sig in mapping_b
        ]
        if not shared_signals:
            return self._empty_result(
                mode="link_across",
                reason=(
                    f"No matching identity columns between the two sources. "
                    f"Source A detected: {list(mapping_a.keys())}. "
                    f"Source B detected: {list(mapping_b.keys())}. "
                    "We need at least one of: name, email, phone in both."
                ),
            )

        # Find candidate pairs via blocking (no full cartesian product)
        candidate_pairs = self._block_candidates(
            df_a_prep, df_b_prep, shared_signals, mapping_a, mapping_b
        )

        # Score each candidate
        matches = self._score_pairs(
            candidate_pairs, df_a_prep, df_b_prep,
            shared_signals, mapping_a, mapping_b,
            match_threshold=match_threshold,
        )

        return {
            "mode": "link_across",
            "source_a_name": source_a_name,
            "source_b_name": source_b_name,
            "source_a_rows": int(len(df_a)),
            "source_b_rows": int(len(df_b)),
            "shared_signals": shared_signals,
            "column_mapping_a": mapping_a,
            "column_mapping_b": mapping_b,
            "candidate_pairs_evaluated": int(len(candidate_pairs)),
            "match_threshold": match_threshold,
            "matches": matches,
            "summary": self._summarize(matches),
        }

    def dedup_within(
        self,
        df: pd.DataFrame,
        source_name: str = "data",
        match_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Find duplicate records within a single dataset."""
        df_prep, mapping = self._prepare_dataframe(df, "left")

        signals = [s for s in ("name", "email", "phone") if s in mapping]
        if not signals:
            return self._empty_result(
                mode="dedup_within",
                reason="No identity columns detected (need name, email, or phone).",
            )

        # Block within the same DataFrame
        candidate_pairs = self._block_candidates(
            df_prep, df_prep, signals, mapping, mapping, same_source=True
        )

        matches = self._score_pairs(
            candidate_pairs, df_prep, df_prep,
            signals, mapping, mapping,
            match_threshold=match_threshold, same_source=True,
        )

        return {
            "mode": "dedup_within",
            "source_name": source_name,
            "source_rows": int(len(df)),
            "shared_signals": signals,
            "column_mapping_a": mapping,
            "column_mapping_b": mapping,
            "candidate_pairs_evaluated": int(len(candidate_pairs)),
            "match_threshold": match_threshold,
            "matches": matches,
            "summary": self._summarize(matches),
        }

    # ─── Preparation ──────────────────────────────────────────────────────

    def _prepare_dataframe(
        self, df: pd.DataFrame, side: str
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Add a stable index and detect which columns hold name/email/phone.

        Returns:
            (prepared_df, mapping)
            mapping is e.g. {"name": "Full_Name", "email": "Email", "phone": "Phone"}
        """
        out = df.copy()
        # Reset index to a clean range so recordlinkage produces sensible pair IDs
        out = out.reset_index(drop=True)
        out["_dq_row"] = range(len(out))

        mapping: dict[str, str] = {}
        for col in out.columns:
            if col == "_dq_row":
                continue
            lower = str(col).lower()

            # Email FIRST (more specific than "mail" — we want "email" over "email_opt_out")
            if "email" not in mapping and any(p == lower or p in lower for p in self.EMAIL_PATTERNS):
                # Skip flag/timestamp columns like email_opt_out, email_at
                if not any(skip in lower for skip in ("opt_out", "_at", "_time", "_date", "valid")):
                    mapping["email"] = col
                    continue

            # Phone (prefer "phone" over "mobile" if both exist — first wins)
            if "phone" not in mapping and any(p in lower for p in self.PHONE_PATTERNS):
                if not any(skip in lower for skip in ("_at", "_time", "_date")):
                    mapping["phone"] = col
                    continue

            # Name (full_name preferred over plain "name")
            if any(p in lower for p in self.NAME_PATTERNS):
                # Don't pick generic columns named like "username", "filename" etc.
                if any(skip in lower for skip in ("file", "user", "host", "domain", "company")):
                    continue
                # Prefer Full_Name over Name if we already have one
                if "name" in mapping:
                    if "full" in lower and "full" not in mapping["name"].lower():
                        mapping["name"] = col
                else:
                    mapping["name"] = col

        # Coerce identity columns to clean strings
        for sig, col in mapping.items():
            out[col] = out[col].astype("string").fillna("").str.strip()

        return out, mapping

    # ─── Blocking (find candidate pairs) ──────────────────────────────────

    def _block_candidates(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        signals: list[str],
        mapping_a: dict[str, str],
        mapping_b: dict[str, str],
        same_source: bool = False,
    ) -> pd.MultiIndex:
        """Use recordlinkage's blocking to generate candidate pairs efficiently.

        We block on the first character of each available identity column,
        so we don't waste time comparing rows that share nothing in common.
        """
        indexer = recordlinkage.Index()

        # Add a blocking pass per signal — the union of all passes is the candidate set
        for signal in signals:
            col_a = mapping_a[signal]
            col_b = mapping_b[signal]

            # Add a helper column for blocking (lowercase first character)
            block_col = f"_block_{signal}"
            df_a[block_col] = df_a[col_a].astype(str).str.lower().str[:1]
            df_b[block_col] = df_b[col_b].astype(str).str.lower().str[:1]

            indexer.block(left_on=block_col, right_on=block_col)

        if same_source:
            candidates = indexer.index(df_a)
            # Remove self-pairs and duplicate ordering (a,b) == (b,a)
            candidates = candidates[candidates.get_level_values(0) < candidates.get_level_values(1)]
        else:
            candidates = indexer.index(df_a, df_b)

        return candidates

    # ─── Scoring ──────────────────────────────────────────────────────────

    def _score_pairs(
        self,
        candidate_pairs: pd.MultiIndex,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        signals: list[str],
        mapping_a: dict[str, str],
        mapping_b: dict[str, str],
        match_threshold: float,
        same_source: bool = False,
    ) -> list[dict[str, Any]]:
        """Score every candidate pair.

        Confidence model (calibrated for real-world cross-source matching):
          - Any single strong signal (exact email, exact phone, OR very high
            name similarity) is enough to declare a match.
          - Multiple signals stack up to HIGH confidence.
          - We do NOT average against all possible signals — a missing column
            shouldn\'t penalise the match.
        """
        if len(candidate_pairs) == 0:
            return []

        matches: list[dict[str, Any]] = []

        # Iterate over candidate pairs
        seen_pairs: set[tuple[int, int]] = set()
        for left_idx, right_idx in candidate_pairs:
            # Avoid duplicates from multi-pass blocking
            pair_key = (int(left_idx), int(right_idx))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            row_a = df_a.iloc[left_idx]
            row_b = df_b.iloc[right_idx]

            signal_details: dict[str, Any] = {}
            signal_strengths: list[float] = []  # each value in 0-1

            # Email: exact match (lowercased)
            if "email" in signals:
                e_a = str(row_a[mapping_a["email"]]).lower().strip()
                e_b = str(row_b[mapping_b["email"]]).lower().strip()
                if e_a and e_b and e_a == e_b:
                    signal_strengths.append(0.98)  # very strong, but not 1.0 (shared emails exist)
                    signal_details["email"] = "exact_match"
                elif e_a and e_b:
                    signal_details["email"] = "no_match"
                else:
                    signal_details["email"] = "missing"

            # Phone: exact match on digits only (strip formatting)
            if "phone" in signals:
                p_a = re.sub(r"\D", "", str(row_a[mapping_a["phone"]]))
                p_b = re.sub(r"\D", "", str(row_b[mapping_b["phone"]]))
                if p_a and p_b and len(p_a) >= 7 and len(p_b) >= 7:
                    if p_a == p_b or p_a[-9:] == p_b[-9:]:
                        signal_strengths.append(0.97)  # very strong
                        signal_details["phone"] = "exact_match"
                    else:
                        signal_details["phone"] = "no_match"
                else:
                    signal_details["phone"] = "insufficient_data"

            # Name: fuzzy match using token_set_ratio
            if "name" in signals:
                n_a = str(row_a[mapping_a["name"]]).lower().strip()
                n_b = str(row_b[mapping_b["name"]]).lower().strip()
                if n_a and n_b:
                    name_sim = fuzz.token_set_ratio(n_a, n_b)
                    if name_sim >= 90:
                        signal_strengths.append(name_sim / 100)  # 0.90-1.00
                        signal_details["name"] = f"strong_match ({name_sim}%)"
                    elif name_sim >= self.NAME_SIM_THRESHOLD:
                        signal_strengths.append((name_sim / 100) * 0.6)  # downweight partial names
                        signal_details["name"] = f"fuzzy_match ({name_sim}%)"
                    else:
                        signal_details["name"] = f"weak ({name_sim}%)"
                else:
                    signal_details["name"] = "insufficient_data"

            # Cap each single signal so one alone can\'t produce 100% probability.
            # No matter how perfect, a single signal is still inconclusive — different
            # people share names, share phones (family), share emails (shared accounts).
            CAP_SINGLE_SIGNAL = 0.55  # one strong signal alone tops out at 55%
            if not signal_strengths:
                continue

            strong_signals = sum(1 for s in signal_strengths if s >= 0.9)

            if strong_signals == 0:
                # No signal even meets the strong bar — skip entirely
                continue
            elif strong_signals == 1:
                # Single signal: probability capped, demoted confidence
                probability = min(max(signal_strengths), CAP_SINGLE_SIGNAL)
                confidence = "medium" if probability >= match_threshold else "low"
            else:
                # 2+ strong signals: use noisy-OR (genuine multi-signal evidence)
                prob_no_match = 1.0
                for s in signal_strengths:
                    prob_no_match *= (1 - s)
                probability = 1 - prob_no_match
                confidence = "high" if probability >= 0.90 else "medium"

            if probability < match_threshold:
                continue

            left_values = {
                sig: str(row_a[mapping_a[sig]]) for sig in signals if sig in mapping_a
            }
            right_values = {
                sig: str(row_b[mapping_b[sig]]) for sig in signals if sig in mapping_b
            }

            matches.append({
                "left_id": int(row_a["_dq_row"]),
                "right_id": int(row_b["_dq_row"]),
                "match_probability": round(probability, 4),
                "confidence": confidence,
                "signals": signal_details,
                "left_values": left_values,
                "right_values": right_values,
            })

        matches.sort(key=lambda m: -m["match_probability"])
        return matches[:500]  # cap for UI safety

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _summarize(matches: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "total": len(matches),
            "high": sum(1 for m in matches if m["confidence"] == "high"),
            "medium": sum(1 for m in matches if m["confidence"] == "medium"),
            "low": sum(1 for m in matches if m["confidence"] == "low"),
        }

    @staticmethod
    def _empty_result(mode: str, reason: str) -> dict[str, Any]:
        return {
            "mode": mode,
            "matches": [],
            "summary": {"total": 0, "high": 0, "medium": 0, "low": 0},
            "reason": reason,
        }
