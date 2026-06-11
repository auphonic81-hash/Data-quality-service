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

            # Short datasets (e.g. single-row PDF extractions) can\'t satisfy
            # the uniqueness check (need 2+ distinct values). If the column has
            # a strong name signal AND a structured value, accept it anyway.
            is_short = n_rows < 5
            qualifies = (
                (has_name and (has_uniqueness or has_format))
                or has_format
                or (is_short and has_name and self._has_id_like_value(df[col]))
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


    def _has_id_like_value(self, series: pd.Series) -> bool:
        """Check if any non-null value looks like a structured ID code.

        Used for short datasets (e.g. single-row PDFs) where the uniqueness
        signal can\'t fire but the value clearly looks like an identifier.
        """
        for v in series.dropna().astype(str).head(10):
            if self._looks_like_id_format(v.strip()):
                return True
        return False

    # ─── Column matching across datasets ──────────────────────────────────

    @staticmethod
    def _find_shared_id_columns(
        dataset_info: list[dict[str, Any]],
    ) -> dict[str, dict[str, str]]:
        """Find ID columns that can be matched across datasets.

        Two-pass strategy:
          Pass 1: Match by lowercase column NAME (fast, handles the easy case)
          Pass 2: For datasets that didn\'t match by name, try matching by VALUE OVERLAP —
                  i.e. "this column in file A and that column in file B share at least 10%
                  of their actual values". Handles Vendor_Code vs vendor_id renames.
        """
        # ── Pass 1: by lowercase name (existing behavior)
        by_canonical: dict[str, dict[str, str]] = defaultdict(dict)
        for d in dataset_info:
            for cand in d["detected_id_columns"]:
                canonical = cand["column"].lower().strip()
                by_canonical[canonical][d["dataset_id"]] = cand["column"]

        # Datasets already matched in Pass 1
        matched_by_name = {c: m for c, m in by_canonical.items() if len(m) >= 2}
        if matched_by_name:
            return matched_by_name

        # ── Pass 2: by value overlap
        # Collect every detected ID column with its actual values
        cols_with_values: list[dict[str, Any]] = []
        for d in dataset_info:
            df = d["dataframe"]
            for cand in d["detected_id_columns"]:
                values = set(df[cand["column"]].dropna().astype(str).str.strip().str.lower())
                if not values:
                    continue
                cols_with_values.append({
                    "dataset_id":  d["dataset_id"],
                    "column":      cand["column"],
                    "values":      values,
                })

        # For every pair of (different-dataset) columns, compute Jaccard overlap.
        # Group columns into shared clusters when overlap >= 10% of smaller set.
        OVERLAP_THRESHOLD = 0.10
        clusters: list[dict[str, dict[str, str]]] = []
        used_keys: set[tuple[str, str]] = set()

        for i in range(len(cols_with_values)):
            ci = cols_with_values[i]
            key_i = (ci["dataset_id"], ci["column"])
            if key_i in used_keys:
                continue
            current_cluster: dict[str, str] = {ci["dataset_id"]: ci["column"]}
            current_values = set(ci["values"])

            for j in range(i + 1, len(cols_with_values)):
                cj = cols_with_values[j]
                if cj["dataset_id"] in current_cluster:
                    continue  # one column per dataset
                shared = len(current_values & cj["values"])
                smaller = min(len(current_values), len(cj["values"]))
                if smaller > 0 and shared / smaller >= OVERLAP_THRESHOLD:
                    current_cluster[cj["dataset_id"]] = cj["column"]
                    used_keys.add((cj["dataset_id"], cj["column"]))

            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
                used_keys.add(key_i)

        # Build the canonical name → per_dataset_col map
        result: dict[str, dict[str, str]] = {}
        for idx, cluster in enumerate(clusters):
            # Pick a canonical name (shortest column name from the cluster)
            canonical = min(cluster.values(), key=len).lower().strip()
            # Avoid collisions if Pass 1 already used this name
            while canonical in result:
                canonical = f"{canonical}_{idx}"
            result[canonical] = cluster

        return result

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


    def suggest_column_groups(
        self,
        datasets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Recommend which columns across files hold the same business concept.

        Strategy:
          1. Compute value overlap between every pair of detected ID columns
             across different datasets (Jaccard on the value sets).
          2. Group columns into clusters where any pair overlaps >= 10%.
          3. Return each cluster with its evidence (overlap %, sample values).

        The caller can then ask an LLM to verify ambiguous clusters.
        """
        # Collect detected ID columns with their value sets
        cols: list[dict[str, Any]] = []
        for d in datasets:
            df = d["dataframe"]
            for cand in self.detect_id_columns(df):
                values = set(df[cand["column"]].dropna().astype(str).str.strip().str.lower())
                if not values:
                    continue
                cols.append({
                    "dataset_id":  d["dataset_id"],
                    "dataset_name": d.get("name", d["dataset_id"]),
                    "column":      cand["column"],
                    "confidence":  cand["confidence"],
                    "sample_values": cand["sample_values"],
                    "uniqueness":  cand["uniqueness"],
                    "values":      values,
                })

        # Pairwise overlap → union-find clustering
        OVERLAP_THRESHOLD = 0.10
        parent = list(range(len(cols)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        pair_overlaps: dict[tuple[int, int], float] = {}
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[i]["dataset_id"] == cols[j]["dataset_id"]:
                    continue
                shared = len(cols[i]["values"] & cols[j]["values"])
                smaller = min(len(cols[i]["values"]), len(cols[j]["values"]))
                if smaller == 0:
                    continue
                overlap = shared / smaller
                pair_overlaps[(i, j)] = overlap
                if overlap >= OVERLAP_THRESHOLD:
                    union(i, j)

        # Build groups
        groups: dict[int, list[int]] = {}
        for i in range(len(cols)):
            groups.setdefault(find(i), []).append(i)

        # Format groups for the UI
        result: list[dict[str, Any]] = []
        for member_indices in groups.values():
            if len(member_indices) < 2:
                continue
            # Compute the best overlap inside this group as evidence
            best_overlap = 0.0
            for i in member_indices:
                for j in member_indices:
                    if i < j:
                        best_overlap = max(best_overlap, pair_overlaps.get((i, j), 0))
            members = []
            for idx in member_indices:
                c = cols[idx]
                members.append({
                    "dataset_id":  c["dataset_id"],
                    "dataset_name": c["dataset_name"],
                    "column":      c["column"],
                    "confidence":  c["confidence"],
                    "sample_values": c["sample_values"],
                    "uniqueness":  c["uniqueness"],
                })
            result.append({
                "group_id":     f"g{len(result)}",
                "members":      members,
                "value_overlap_pct": round(best_overlap * 100, 1),
                "auto_recommended": best_overlap >= 0.30,
            })

        # Sort: auto-recommended groups first, then by overlap %
        result.sort(key=lambda g: (not g["auto_recommended"], -g["value_overlap_pct"]))
        return result


    def classify_groups(
        self,
        groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Classify each group as PK match / FK link / uncertain using heuristics.

        Rules:
          - PK-like name (*_no, *_number, *_ref, invoice_*, bill_*, order_*) +
            overlap between 5% and 60%  →  primary_key_match
          - FK-like name (customer_id, vendor_id, supplier_id, product_id) +
            overlap > 60%  →  foreign_key_link
          - Otherwise → uncertain (user picks manually)
        """
        PK_HINTS = ("_no", "_number", "_ref", "invoice", "bill", "order", "receipt", "transaction", "txn")
        FK_HINTS = ("customer", "client", "vendor", "supplier", "product", "user", "account")

        for group in groups:
            overlap_pct = group["value_overlap_pct"]
            # Take the column-name signal from any member (they all map to the same concept)
            sample_col = group["members"][0]["column"].lower()

            is_pk_name = any(h in sample_col for h in PK_HINTS)
            is_fk_name = any(h in sample_col for h in FK_HINTS) and sample_col.endswith(("_id", "id"))

            if is_pk_name and 5 <= overlap_pct <= 60:
                cls = "primary_key_match"
                reason = f"Column name suggests a record key ({sample_col}) and {overlap_pct}% overlap is consistent with partial duplication."
            elif is_fk_name and overlap_pct > 60:
                cls = "foreign_key_link"
                reason = f"Column name looks like a foreign key ({sample_col}) and {overlap_pct}% overlap is consistent with shared references — not duplicates."
            elif overlap_pct > 90:
                cls = "foreign_key_link"
                reason = f"{overlap_pct}% value overlap is too high for a primary key — usually indicates a shared reference."
            elif overlap_pct < 5:
                cls = "uncertain"
                reason = f"Only {overlap_pct}% overlap — these columns may not represent the same concept."
            else:
                cls = "uncertain"
                reason = "Cannot determine automatically — please decide based on the sample values."

            group["classification"] = cls
            group["reasoning"] = reason
            group["dedup_recommended"] = (cls == "primary_key_match")

        order = {"primary_key_match": 0, "uncertain": 1, "foreign_key_link": 2}
        groups.sort(key=lambda g: order.get(g.get("classification", "uncertain"), 1))
        return groups

    def classify_groups_with_llm(
        self,
        groups: list[dict[str, Any]],
        ollama_url: str = "http://localhost:11434/api/generate",
        model: str = "qwen3.5:latest",
    ) -> list[dict[str, Any]]:
        """Ask the local LLM to classify each group as PK match vs FK link.

        Adds three fields to each group:
          - classification: "primary_key_match" | "foreign_key_link" | "uncertain"
          - reasoning: short explanation
          - dedup_recommended: bool (true iff PK match)

        Falls back to "uncertain" with a clear note if the LLM is unreachable.
        """
        import json as _json
        import requests

        for group in groups:
            # Strip the heavy "values" set before sending to LLM
            payload_members = [
                {
                    "dataset": m["dataset_name"],
                    "column":  m["column"],
                    "samples": m["sample_values"][:5],
                    "uniqueness": m["uniqueness"],
                }
                for m in group["members"]
            ]
            value_overlap = group["value_overlap_pct"]

            prompt = f"""You are a database expert classifying column relationships between files.

Given these matching columns across files:
{_json.dumps(payload_members, indent=2)}

Value overlap across files: {value_overlap}% of values appear in both.

Classify this relationship as exactly ONE of:
- "primary_key_match": Same record appearing in both files (true duplicate). Column holds the record\'s own unique ID. Example: same invoice number in orders.csv and invoices.csv → these rows represent the same bill.
- "foreign_key_link": Column references a record in another table. High overlap is expected and normal — NOT a sign of duplication. Example: customer_id in orders.csv and invoices.csv → same customer placing orders AND being invoiced is not a duplicate.
- "uncertain": Cannot determine from the signal.

Key clues:
- If overlap is near 100% and the column looks like *_id pointing to entities (customers, products, vendors), it\'s usually a foreign key.
- If overlap is partial (10-30%) and the column looks like the file\'s own unique record key (bill_no, invoice_ref, order_id where each file is about orders/invoices), it\'s a primary key match.
- Look at the column name semantics and the sample values.

Return ONLY valid JSON, no other text:
{{"classification": "...", "reasoning": "one short sentence"}}
"""

            try:
                response = requests.post(
                    ollama_url,
                    json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
                    timeout=60,
                )
                response.raise_for_status()
                body = response.json()
                # Reasoning models (qwen 3.5) put the answer in "thinking" when
                # the model didn\'t produce a separate "response" field.
                raw = body.get("response") or body.get("thinking") or "{}"
                if isinstance(raw, str):
                    raw = raw.strip()
                    # Strip markdown code fences if present
                    if raw.startswith("```"):
                        raw = raw.strip("`").strip()
                        if raw.lower().startswith("json"):
                            raw = raw[4:].strip()
                    parsed = _json.loads(raw)
                else:
                    parsed = raw
                cls = parsed.get("classification", "uncertain")
                if cls not in ("primary_key_match", "foreign_key_link", "uncertain"):
                    cls = "uncertain"
                group["classification"] = cls
                group["reasoning"] = parsed.get("reasoning", "")
                group["dedup_recommended"] = (cls == "primary_key_match")
            except Exception as exc:
                group["classification"] = "uncertain"
                group["reasoning"] = f"LLM unavailable ({type(exc).__name__}). Pick manually."
                group["dedup_recommended"] = False

        # Re-sort: PK matches first, uncertain next, FK last
        order = {"primary_key_match": 0, "uncertain": 1, "foreign_key_link": 2}
        groups.sort(key=lambda g: order.get(g.get("classification", "uncertain"), 1))
        return groups

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
