"""Master Data Repository — the unified data platform.

This is the central concept the owner asked for:
  - Sources (CSV, Excel, PDF, future: SAP/SharePoint) feed INTO this repository
  - Four master tables: Customers, Vendors, Products, Invoices
  - Every ingestion runs the full pipeline automatically
  - Lineage is tracked per row: which source contributed it
  - Duplicates are merged (golden record), not duplicated

Inspired by Informatica MDM but built on SQLite for local demonstration.
A production deployment would use Postgres or similar.
"""
from __future__ import annotations

import sqlite3
import uuid
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import pandas as pd


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


import re as _re
_YEAR_PAT = _re.compile(r"(?<![0-9])(19|20)\d{2}(?![0-9])")
_SEP_PAT  = _re.compile(r"[\s\-_/.]+")

def normalize_key(value):
    """Canonicalize an identifier for cross-source matching."""
    if value is None:
        return ""
    s = str(value).strip().upper()
    if not s:
        return ""
    s = _YEAR_PAT.sub("", s)
    s = _SEP_PAT.sub("-", s)
    s = s.strip("-")
    parts = s.split("-")
    cleaned = []
    for p in parts:
        if p.isdigit():
            cleaned.append(str(int(p)) if p else p)
        else:
            cleaned.append(p)
    return "-".join(c for c in cleaned if c)



# ─── Column mappings ────────────────────────────────────────────────────
# Maps common source column names to canonical master schema columns.
# When a source has "Full_Name", "name", or "customer_name", they all map to
# the master "full_name" column.

CUSTOMER_FIELD_MAP = {
    "customer_id": ["customer_id", "cust_id", "id", "client_id", "crm_id"],
    "full_name":   ["full_name", "name", "customer_name", "client_name", "fullname"],
    "email":       ["email", "mail", "e_mail", "email_address"],
    "phone":       ["phone", "mobile", "telephone", "cell", "contact_phone"],
    "city":        ["city", "town"],
    "country":     ["country", "country_name", "country_code", "nation"],
    "address":     ["address", "addressline1", "street", "address_line_1"],
    "status":      ["status", "state", "active"],
    "created_at":  ["created_at", "created", "registration_date", "signup_date"],
}

VENDOR_FIELD_MAP = {
    "vendor_id":     ["vendor_id", "vendor_code", "supplier_id", "id"],
    "vendor_name":   ["vendor_name", "name", "company_name", "supplier_name"],
    "contact_phone": ["contact_phone", "phone", "telephone", "mobile"],
    "country":       ["country", "country_name", "nation"],
    "status":        ["status", "active", "state"],
}

PRODUCT_FIELD_MAP = {
    "sku":          ["sku", "product_code", "item_code", "product_sku"],
    "product_name": ["product_name", "name", "title", "description"],
    "category":     ["category", "product_category", "type", "product_line"],
    "price":        ["price", "unit_price", "cost"],
    "msrp":         ["msrp", "list_price", "retail_price"],
    "supplier_id":  ["supplier_id", "vendor_id", "supplier"],
    "stock":        ["stock", "quantity", "inventory", "qty"],
}

INVOICE_FIELD_MAP = {
    "invoice_no":   ["invoice_no", "invoice_number", "bill_no", "order_number", "receipt_no", "invoice_id"],
    "customer_id":  ["customer_id", "client_id", "cust_id"],
    "amount":       ["amount", "total", "line_total", "total_amount", "value"],
    "due_date":     ["due_date", "due", "payment_due"],
    "invoice_date": ["invoice_date", "issue_date", "order_date", "date"],
    "status":       ["status", "payment_status", "state"],
    "description":  ["description", "notes", "memo", "item_description"],
}

MASTER_TABLES = {
    "Customers": ("master_customers", "customer_id", CUSTOMER_FIELD_MAP),
    "Vendors":   ("master_vendors",   "vendor_id",   VENDOR_FIELD_MAP),
    "Products":  ("master_products",  "sku",         PRODUCT_FIELD_MAP),
    "Invoices":  ("master_invoices",  "invoice_no",  INVOICE_FIELD_MAP),
}


class MasterRepository:
    """The unified data platform — ingests sources, maintains golden records."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)

    # ─── Public API ──────────────────────────────────────────────────────

    def ingest(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        target_entity: str,
    ) -> dict[str, Any]:
        """Ingest a DataFrame into the specified master entity.

        Pipeline:
          1. Map source columns → master schema columns
          2. For each row: look up by primary key in master table
             - If found: merge (update master with newer/more-complete fields)
             - If not found: insert as new master row
          3. Log everything to ingestion_log

        Returns summary of what happened.
        """
        if target_entity not in MASTER_TABLES:
            raise ValueError(
                f"Unknown entity '{target_entity}'. "
                f"Must be one of: {list(MASTER_TABLES)}"
            )

        table_name, pk_column, field_map = MASTER_TABLES[target_entity]

        # Step 1 — map source columns to canonical master columns
        mapped_df, mapping_used = self._map_columns(df, field_map)
        if pk_column not in mapped_df.columns:
            return {
                "success": False,
                "error": f"Could not find a {pk_column} column in the source. "
                         f"Tried: {field_map[pk_column]}",
                "mapping_used": mapping_used,
            }

        # Step 2 — ingest each row
        rows_inserted = 0
        rows_duplicates_found = 0
        rows_enriched = 0
        rows_rejected = 0
        fields_enriched_total = 0
        duplicate_details: list[dict[str, Any]] = []
        now = _now()
        ingestion_id = uuid.uuid4().hex[:12]

        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            for row_idx, row in mapped_df.iterrows():
                pk_value = row.get(pk_column)
                if pd.isna(pk_value) or str(pk_value).strip() == "":
                    rows_rejected += 1
                    continue
                pk_value = str(pk_value).strip()

                # Build the master record dict
                master_record = {}
                for col in mapped_df.columns:
                    val = row[col]
                    if pd.isna(val):
                        continue
                    master_record[col] = val

                # Check if this PK already exists — exact match first, then normalized.
                existing = con.execute(
                    f"SELECT * FROM {table_name} WHERE {pk_column}=? AND archived_at IS NULL",
                    (pk_value,),
                ).fetchone()
                if not existing:
                    norm = normalize_key(pk_value)
                    if norm:
                        for c in con.execute(f"SELECT * FROM {table_name} WHERE archived_at IS NULL"):
                            if normalize_key(c[pk_column]) == norm:
                                existing = c
                                break

                if existing:
                    # DUPLICATE FOUND — enrich master from source, archive the source row
                    merge_result = self._merge_row(
                        con, table_name, pk_column, pk_value,
                        existing, master_record, dataset_id, row_idx, now,
                    )
                    rows_duplicates_found += 1
                    if merge_result["fields_enriched"] > 0:
                        rows_enriched += 1
                        fields_enriched_total += merge_result["fields_enriched"]
                    duplicate_details.append({
                        "source_row_index": int(row_idx),
                        "matched_pk":       pk_value,
                        "fields_enriched":  merge_result["fields_enriched"],
                        "enriched_columns": merge_result["enriched_columns"],
                    })
                else:
                    # INSERT new master row
                    self._insert_row(
                        con, table_name, master_record, dataset_id,
                        row_idx, now, pk_column, pk_value,
                    )
                    rows_inserted += 1

            # Log the ingestion (rows_merged column repurposed to mean "duplicates found")
            con.execute(
                "INSERT INTO ingestion_log "
                "(ingestion_id, dataset_id, target_table, rows_inserted, "
                "rows_merged, rows_rejected, fields_standardized, ingested_at, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ingestion_id, dataset_id, target_entity, rows_inserted,
                 rows_duplicates_found, rows_rejected, fields_enriched_total, now,
                 json.dumps({
                     "mapping_used": mapping_used,
                     "rows_enriched": rows_enriched,
                     "duplicate_details": duplicate_details[:20],  # cap stored detail
                 })),
            )
            con.commit()

        return {
            "success": True,
            "ingestion_id":   ingestion_id,
            "target_entity":  target_entity,
            "target_table":   table_name,
            "rows_inserted":  rows_inserted,
            "rows_duplicates_found": rows_duplicates_found,
            "rows_enriched":  rows_enriched,
            "fields_enriched_total": fields_enriched_total,
            "rows_rejected":  rows_rejected,
            "mapping_used":   mapping_used,
            "duplicate_details": duplicate_details[:10],  # first 10 for UI
            "total_in_master": self._table_count(target_entity),
        }

    def get_master_summary(self) -> dict[str, Any]:
        """Return the unified-repository summary for the dashboard landing view."""
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            summary = {}
            for entity, (table, _, _) in MASTER_TABLES.items():
                row = con.execute(
                    f"SELECT COUNT(*) AS rows, "
                    f"COUNT(DISTINCT source_dataset_id) AS sources, "
                    f"SUM(merged_from_count) AS total_inputs "
                    f"FROM {table} WHERE archived_at IS NULL"
                ).fetchone()
                summary[entity] = {
                    "rows":           row["rows"] or 0,
                    "sources":        row["sources"] or 0,
                    "total_inputs":   row["total_inputs"] or 0,
                    "duplicates_merged": (row["total_inputs"] or 0) - (row["rows"] or 0),
                }
            # Overall totals
            total_sources = con.execute(
                "SELECT COUNT(DISTINCT dataset_id) FROM ingestion_log"
            ).fetchone()[0]
            total_ingestions = con.execute(
                "SELECT COUNT(*) FROM ingestion_log"
            ).fetchone()[0]
            return {
                "entities":         summary,
                "total_sources":    total_sources,
                "total_ingestions": total_ingestions,
            }

    def list_master_rows(self, entity: str, limit: int = 50) -> list[dict[str, Any]]:
        """List rows from a master table (for the dashboard 'browse' view)."""
        if entity not in MASTER_TABLES:
            return []
        table, _, _ = MASTER_TABLES[entity]
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f"SELECT * FROM {table} WHERE archived_at IS NULL "
                f"ORDER BY last_updated DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_ingestion_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the recent ingestion history (for the lineage view)."""
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT i.*, d.filename FROM ingestion_log i "
                "LEFT JOIN datasets d ON d.dataset_id = i.dataset_id "
                "ORDER BY i.ingested_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _map_columns(
        df: pd.DataFrame, field_map: dict[str, list[str]]
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Rename source columns to canonical master columns. Case-insensitive."""
        lower_to_real = {str(c).lower(): str(c) for c in df.columns}
        rename = {}
        mapping_used = {}
        for canonical, candidates in field_map.items():
            for cand in candidates:
                if cand.lower() in lower_to_real:
                    real = lower_to_real[cand.lower()]
                    rename[real] = canonical
                    mapping_used[canonical] = real
                    break
        return df.rename(columns=rename), mapping_used

    def _insert_row(
        self, con, table: str, record: dict[str, Any], dataset_id: str,
        row_idx: int, now: str, pk_column: str, pk_value: str,
    ) -> None:
        record[pk_column] = pk_value
        record["source_dataset_id"] = dataset_id
        record["source_row_index"]  = int(row_idx)
        record["ingested_at"]       = now
        record["last_updated"]      = now
        record["merged_from_count"] = 1
        record["confidence_score"]  = 1.0
        # Only keep columns that exist in the target table
        existing_cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        filtered = {k: v for k, v in record.items() if k in existing_cols}
        cols = ", ".join(filtered.keys())
        placeholders = ", ".join("?" * len(filtered))
        con.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            tuple(filtered.values()),
        )

    @staticmethod
    def _merge_row(
        con, table: str, pk_column: str, pk_value: str,
        existing: sqlite3.Row, new_record: dict[str, Any],
        dataset_id: str, row_idx: int, now: str,
    ) -> dict[str, Any]:
        """Auto-enrich the master from a duplicate source row.

        Rule: this is a duplicate — the master record wins for identity, but we
        pick up any new fields the source has that the master was missing.
        Never overwrite established values. Always bump merged_from_count.

        Returns: {"fields_enriched": int, "enriched_columns": [str, ...]}
        """
        existing_dict = dict(existing)
        updates: dict[str, Any] = {}
        for k, v in new_record.items():
            if k not in existing_dict:
                continue
            current = existing_dict[k]
            if (current is None or str(current).strip() == "") and v is not None and str(v).strip() != "":
                updates[k] = v

        # Always bump the merge counter (this PK confirmed by another source)
        updates["last_updated"] = now
        updates["merged_from_count"] = (existing_dict.get("merged_from_count") or 1) + 1

        set_clause = ", ".join(f"{k}=?" for k in updates.keys())
        con.execute(
            f"UPDATE {table} SET {set_clause} WHERE {pk_column} = ?",
            (*updates.values(), pk_value),
        )
        # Don\'t count the counter columns as "enriched" — those are bookkeeping
        enriched = [k for k in updates if k not in ("last_updated", "merged_from_count")]
        return {
            "fields_enriched": len(enriched),
            "enriched_columns": enriched,
        }

    def _table_count(self, entity: str) -> int:
        table, _, _ = MASTER_TABLES[entity]
        with sqlite3.connect(self.db_path) as con:
            return con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE archived_at IS NULL"
            ).fetchone()[0]
