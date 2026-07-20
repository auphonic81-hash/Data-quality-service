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
    # Preserve year and digit length: INV-2024-00200 != INV-000200
    s = _SEP_PAT.sub("-", s)
    s = s.strip("-")
    return s



# ─── Column mappings ────────────────────────────────────────────────────
# Maps common source column names to canonical master schema columns.
# When a source has "Full_Name", "name", or "customer_name", they all map to
# the master "full_name" column.

CUSTOMER_FIELD_MAP = {
    "customer_id": ["customer_id", "cust_id", "cust_no", "custno", "id", "client_id", "crm_id"],
    "full_name":   ["full_name", "name", "name1", "customer_name", "client_name", "fullname"],
    "email":       ["email", "mail", "e_mail", "email_address", "email_adr"],
    "phone":       ["phone", "mobile", "telephone", "cell", "contact_phone", "phone_nr"],
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
    "sku":          ["sku", "product_code", "item_code", "product_sku", "prod_code"],
    "product_name": ["product_name", "name", "title", "description"],
    "category":     ["category", "product_category", "type", "product_line"],
    "price":        ["price", "unit_price", "cost"],
    "msrp":         ["msrp", "list_price", "retail_price"],
    "supplier_id":  ["supplier_id", "vendor_id", "supplier"],
    "stock":        ["stock", "quantity", "inventory", "qty"],
}

INVOICE_FIELD_MAP = {
    "invoice_no":   ["invoice_no", "invoice_number", "bill_no", "bill_nr", "order_number", "receipt_no", "invoice_id"],
    "customer_id":  ["customer_id", "client_id", "cust_id"],
    "amount":       ["amount", "total", "line_total", "total_amount", "value", "amt"],
    "due_date":     ["due_date", "due", "payment_due"],
    "invoice_date": ["invoice_date", "issue_date", "order_date", "date"],
    "status":       ["status", "payment_status", "state"],
    "description":  ["description", "notes", "memo", "item_description"],
    "bill_to_name": ["bill_to_name", "billed_to", "customer_name"],
    "vendor": ["vendor", "vendor_name", "supplier", "from"],
    "line_item_count": ["line_item_count"],
    "line_items_sum": ["line_items_sum"],
    "line_items_json": ["line_items_json"],
}

MASTER_TABLES = {
    "Customers": ("master_customers", "customer_id", CUSTOMER_FIELD_MAP),
    "Vendors":   ("master_vendors",   "vendor_id",   VENDOR_FIELD_MAP),
    "Products":  ("master_products",  "sku",         PRODUCT_FIELD_MAP),
    "Invoices":  ("master_invoices",  "invoice_no",  INVOICE_FIELD_MAP),
}


from .openlineage_emitter import OpenLineageEmitter
from .llm_mapper import LLMColumnMapper

EMPLOYEE_FIELD_MAP = {
    "employee_id":     ["employee_id", "emp_id", "emp_no", "empno", "empid", "id"],
    "full_name":       ["full_name", "name", "full_nm", "fullname", "employee_name"],
    "email":           ["email", "mail", "work_email", "email_address", "e_mail"],
    "phone":           ["phone", "mobile", "mobile_nr", "telephone", "phone_nr", "cell"],
    "department_code": ["department_code", "dept_cd", "dept_code", "department_id"],
    "department_name": ["department_name", "dept_nm", "dept_name", "department"],
    "country":         ["country", "ctry", "land", "land_code", "country_code"],
    "hire_date":       ["hire_date", "hire_dt", "start_date", "joined", "crdat"],
    "status":          ["status", "state", "employment_status"],
    "manager_id":      ["manager_id", "mgr_id", "manager", "reports_to"],
    "salary":          ["salary", "salary_usd", "annual_salary", "comp", "pay"],
}

# Register Employees in MASTER_TABLES (placed here because EMPLOYEE_FIELD_MAP must exist first)
MASTER_TABLES["Employees"] = ("master_employees", "employee_id", EMPLOYEE_FIELD_MAP)

# ===========================================================================
# Extended entities (added by 14-entity patch)
# ===========================================================================

ACCOUNT_FIELD_MAP = {
    "acct_id":         ["acct_id", "account_id", "acc_id", "ACCT_ID"],
    "account_name":    ["account_name", "acct_name", "name", "company_name"],
    "annual_revenue":  ["annual_revenue", "revenue", "yearly_revenue", "arr"],
    "employee_count":  ["employee_count", "emp_count", "headcount", "num_employees"],
    "industry":        ["industry", "sector", "vertical"],
    "country":         ["country", "ctry", "country_code"],
    "status":          ["status", "state", "account_status"],
}
MASTER_TABLES["Accounts"] = ("master_accounts", "acct_id", ACCOUNT_FIELD_MAP)

CONTRACT_FIELD_MAP = {
    "contract_id":    ["contract_id", "ct_id", "CONTRACT_ID", "contract_no"],
    "acct_id":        ["acct_id", "account_id", "customer_acct_id"],
    "contract_value": ["contract_value", "value", "amount", "contract_amt"],
    "start_dt":       ["start_dt", "start_date", "started", "effective_date"],
    "end_dt":         ["end_dt", "end_date", "expires", "expiration_date"],
    "status":         ["status", "contract_status", "state"],
}
MASTER_TABLES["Contracts"] = ("master_contracts", "contract_id", CONTRACT_FIELD_MAP)

DEPARTMENT_FIELD_MAP = {
    "department_code": ["department_code", "dept_code", "dept_cd", "dept_id"],
    "dept_name":       ["dept_name", "department_name", "name"],
    "cost_center":     ["cost_center", "cc", "cost_cd", "cost_centre"],
    "manager_emp_id":  ["manager_emp_id", "manager_id", "manager", "mgr_id"],
}
MASTER_TABLES["Departments"] = ("master_departments", "department_code", DEPARTMENT_FIELD_MAP)

INVENTORY_FIELD_MAP = {
    "inv_item_id":   ["inv_item_id", "item_id", "inventory_id", "stock_id"],
    "sku":           ["sku", "product_sku", "product_id"],
    "loc_id":        ["loc_id", "location_id", "warehouse_id"],
    "quantity":      ["quantity", "qty", "stock", "on_hand"],
    "unit_cost":     ["unit_cost", "cost", "cost_price"],
    "last_count_dt": ["last_count_dt", "last_counted", "count_date"],
}
MASTER_TABLES["Inventory"] = ("master_inventory", "inv_item_id", INVENTORY_FIELD_MAP)

LOCATION_FIELD_MAP = {
    "loc_id":  ["loc_id", "location_id", "site_id", "warehouse_id"],
    "city":    ["city", "town"],
    "country": ["country", "ctry", "country_code"],
    "acct_id": ["acct_id", "account_id"],
    "address": ["address", "street", "addr"],
}
MASTER_TABLES["Locations"] = ("master_locations", "loc_id", LOCATION_FIELD_MAP)

OPPORTUNITY_FIELD_MAP = {
    "opp_id":       ["opp_id", "opportunity_id", "deal_id"],
    "opp_name":     ["opp_name", "opportunity_name", "deal_name", "name"],
    "acct_id":      ["acct_id", "account_id"],
    "owner_emp_id": ["owner_emp_id", "owner", "owner_id", "sales_rep", "rep_id"],
    "amount":       ["amount", "value", "deal_size", "opp_amount"],
    "status":       ["status", "stage", "opp_stage"],
    "close_dt":     ["close_dt", "close_date", "expected_close", "closed_dt"],
}
MASTER_TABLES["Opportunities"] = ("master_opportunities", "opp_id", OPPORTUNITY_FIELD_MAP)

PAYMENT_FIELD_MAP = {
    "pmt_id":       ["pmt_id", "payment_id", "pay_id"],
    "pmt_amt":      ["pmt_amt", "amount", "payment_amount", "pay_amt"],
    "reference_no": ["reference_no", "ref_no", "ref", "reference"],
    "pmt_dt":       ["pmt_dt", "payment_date", "paid_dt", "pay_date"],
    "invoice_no":   ["invoice_no", "invoice_id", "bill_no"],
    "customer_id":  ["customer_id", "cust_id"],
    "method":       ["method", "payment_method", "pay_method"],
}
MASTER_TABLES["Payments"] = ("master_payments", "pmt_id", PAYMENT_FIELD_MAP)

SHIPMENT_FIELD_MAP = {
    "ship_id":     ["ship_id", "shipment_id", "shp_id"],
    "invoice_no":  ["invoice_no", "order_no", "order_id"],
    "tracking_no": ["tracking_no", "tracking", "track_id", "tracking_number"],
    "shipped_dt":  ["shipped_dt", "ship_date", "shipment_date"],
    "status":      ["status", "shipment_status"],
    "carrier":     ["carrier", "carrier_name", "shipper"],
}
MASTER_TABLES["Shipments"] = ("master_shipments", "ship_id", SHIPMENT_FIELD_MAP)

SALESORDER_FIELD_MAP = {
    "order_id":     ["order_id", "order_number", "order_no", "ord_id", "ORDER_ID"],
    "customer_id":  ["customer_id", "cust_id"],
    "product_id":   ["product_id", "product_sku", "sku", "PRODUCT_ID"],
    "amount":       ["amount", "unit_price", "price"],
    "quantity":     ["quantity", "qty", "QTY"],
    "total_amt":    ["total_amt", "total", "total_amount", "TOTAL_AMT"],
    "order_date":   ["order_date", "ord_date", "date", "invoice_date"],
    "status":       ["status", "order_status"],
}
MASTER_TABLES["SalesOrders"] = ("master_sales_orders", "order_id", SALESORDER_FIELD_MAP)



class MasterRepository:
    """The unified data platform — ingests sources, maintains golden records."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.openlineage = OpenLineageEmitter(self.db_path)
        self.llm_mapper  = LLMColumnMapper(self.db_path)

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
        mapped_df, mapping_used = self._map_columns(df, field_map, target_entity=target_entity)

        # Step 1a — entity-aware status vocabulary translation
        # Must run AFTER _map_columns because it operates on the canonical
        # 'status' column. The remediator handles syntax (case); this
        # handles semantics (legacy ERP 'Active'/'Inactive' -> 'Open'/'Paid').
        _ENTITY_STATUS_VOCAB = {
            "Invoices": {"Active": "Open", "Inactive": "Paid"},
            "Shipments": {"Active": "Shipped", "Inactive": "Delivered"},
        }
        _vocab = _ENTITY_STATUS_VOCAB.get(target_entity)
        if _vocab and "status" in mapped_df.columns:
            mapped_df["status"] = mapped_df["status"].replace(_vocab)

        # Step 1b — schema validation via pandera (rejects bad rows BEFORE master)
        rejected_count = 0
        try:
            from .schemas import validate as _validate
            valid_df, rejected_df, _errors = _validate(mapped_df, target_entity)
            if not rejected_df.empty:
                rejected_count = len(rejected_df)
                # Persist rejections for audit — JSON blob schema accepts ANY entity
                import sqlite3 as _sql, json as _json
                _con = _sql.connect(self.db_path, timeout=30)
                try:
                    _con.execute("""CREATE TABLE IF NOT EXISTS master_rejections_v2 (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity TEXT, dataset_id TEXT, pk_value TEXT,
                        rejection_reason TEXT, row_data TEXT, rejected_at TEXT
                    )""")
                    _pk_col = pk_column
                    for _, _row in rejected_df.iterrows():
                        _con.execute(
                            "INSERT INTO master_rejections_v2 (entity, dataset_id, pk_value, rejection_reason, row_data, rejected_at) VALUES (?,?,?,?,?,?)",
                            (target_entity, dataset_id,
                             str(_row.get(_pk_col, "")) if _pk_col in _row.index else "",
                             str(_row.get("_rejection_reason", "")),
                             _json.dumps({k: str(v) for k, v in _row.items() if not k.startswith("_")}),
                             _now())
                        )
                    _con.commit()
                finally:
                    _con.close()
            mapped_df = valid_df
        except ImportError:
            pass  # pandera not installed — skip validation
        except Exception as _ex:
            print(f"[pandera] WARN: {_ex}")

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

        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row

            # Auto-create master table if it doesn't exist yet.
            # New entities registered in MASTER_TABLES don't get init'd at
            # startup; we create on first insert using field_map keys as columns.
            # All columns are TEXT (SQLite is permissive); audit columns added.
            _cols_sql = ", ".join(
                f'"{c}" TEXT' for c in field_map.keys()
            )
            con.execute(
                f'CREATE TABLE IF NOT EXISTS {table_name} ('
                f'{_cols_sql}, '
                f'created_at TEXT, updated_at TEXT, archived_at TEXT'
                f')'
            )

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
                        # Build normalized-key index once per ingestion call (local var, not connection attr)
                        if "_norm_idx" not in locals():
                            _norm_idx = {}
                            for c in con.execute(f"SELECT * FROM {table_name} WHERE archived_at IS NULL"):
                                n = normalize_key(c[pk_column])
                                if n:
                                    _norm_idx[n] = c
                        existing = _norm_idx.get(norm)

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
                        "field_changes":    merge_result.get("field_changes", []),
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

        # Emit OpenLineage events for this run
        try:
            self.openlineage.emit_ingest_run(
                source_dataset_id=dataset_id,
                source_name=str(dataset_id),
                target_master_table=target_entity,
                rows_inserted=rows_inserted,
                rows_duplicates_found=rows_duplicates_found,
                fields_enriched_total=fields_enriched_total,
            )
        except Exception:
            pass  # never break ingest on lineage error

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
            "rows_rejected_by_schema": rejected_count,
            "mapping_used":   mapping_used,
            "duplicate_details": duplicate_details[:10],  # first 10 for UI
            "total_in_master": self._table_count(target_entity),
        }

    def get_master_summary(self) -> dict[str, Any]:
        """Return the unified-repository summary for the dashboard landing view.

        Scans ALL master_* tables dynamically — including the auto-promoted ones —
        not just the 5 hardcoded entities. Robust to tables missing 'archived_at'
        or 'source_dataset_id' (the auto-promoted ones don\'t have full audit cols).
        """
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row
            summary = {}
            # Find every master_* table (skip the record_sources join table)
            master_tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'master_%' AND name NOT IN ('master_record_sources', 'master_rejections_v2') "
                "ORDER BY name"
            ).fetchall()]
            for table in master_tables:
                entity = table.replace("master_", "").capitalize()
                # Detect which audit columns exist on this specific table
                cols = {c[1] for c in con.execute(f"PRAGMA table_info({table})").fetchall()}
                where = " WHERE archived_at IS NULL" if "archived_at" in cols else ""
                # Sources count: from master_record_sources (which tracks ALL contributors),
                # not from the source_dataset_id column (which only records the INITIAL inserter).
                src_expr = ("(SELECT COUNT(DISTINCT substr(source_system, 1, instr(source_system || '_', '_') - 1)) "
                            "FROM master_record_sources WHERE master_table='" + table + "')")
                merged_expr = "SUM(merged_from_count)" if "merged_from_count" in cols else "COUNT(*)"
                try:
                    row = con.execute(
                        f"SELECT COUNT(*) AS rows, {src_expr} AS sources, "
                        f"{merged_expr} AS total_inputs FROM {table}{where}"
                    ).fetchone()
                    summary[entity] = {
                        "rows":              row["rows"] or 0,
                        "sources":           row["sources"] or 0,
                        "total_inputs":      row["total_inputs"] or 0,
                        "duplicates_merged": (row["total_inputs"] or 0) - (row["rows"] or 0),
                    }
                except Exception as exc:
                    summary[entity] = {"rows": 0, "sources": 0, "total_inputs": 0,
                                       "duplicates_merged": 0, "error": str(exc)}
            # Overall totals
            total_sources = con.execute(
                "SELECT COUNT(DISTINCT substr(source_system, 1, instr(source_system || '_', '_') - 1)) FROM datasets WHERE source_system IS NOT NULL"
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
        """List rows from a master table (for the dashboard 'browse' view).

        Dynamic: works for ANY entity that has a master_<entity> table in the DB.
        Robust to tables missing 'archived_at' or 'last_updated' columns
        (auto-promoted entities don\'t have full audit columns).
        """
        table = f"master_{entity.lower()}"
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row
            # Verify table exists
            exists = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            ).fetchone()
            if not exists:
                return []
            # Detect available audit columns
            cols = {c[1] for c in con.execute(f"PRAGMA table_info({table})").fetchall()}
            where  = " WHERE archived_at IS NULL" if "archived_at" in cols else ""
            order  = " ORDER BY last_updated DESC" if "last_updated" in cols else ""
            try:
                rows = con.execute(
                    f"SELECT * FROM {table}{where}{order} LIMIT ?", (limit,)
                ).fetchall()
                # Strip internal audit columns (anything starting with _)
                return [
                    {k: v for k, v in dict(r).items() if not k.startswith("_")}
                    for r in rows
                ]
            except Exception:
                return []

    def get_ingestion_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the recent ingestion history (for the lineage view)."""
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT i.*, d.filename FROM ingestion_log i "
                "LEFT JOIN datasets d ON d.dataset_id = i.dataset_id "
                "ORDER BY i.ingested_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_record_sources(self, entity: str, pk_value: str) -> list[dict[str, Any]]:
        """Return every source that contributed to a specific master record.
        Reads directly from master_record_sources, which has one row per
        (master_table, pk_value, source_system, dataset_id) tuple.
        The drawer's LINEAGE TIMELINE uses this instead of walking ingestion_log.
        """
        table_name = "master_" + entity.lower()
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT s.source_system, s.dataset_id, s.first_seen_at, "
                "       d.filename, d.source "
                "FROM master_record_sources s "
                "LEFT JOIN datasets d ON d.dataset_id = s.dataset_id "
                "WHERE s.master_table = ? AND s.pk_value = ? "
                "ORDER BY s.first_seen_at",
                (table_name, str(pk_value)),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── Internal helpers ────────────────────────────────────────────────

    def _map_columns(
        self,
        df: pd.DataFrame,
        field_map: dict[str, list[str]],
        target_entity: str = None,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        """Rename source columns to canonical master columns.

        Strategy:
          1. Fast path — exact alias match (case-insensitive)
          2. LLM fallback — for any source column NOT mapped via aliases,
             ask the local LLM if it matches an unfilled canonical field.
        """
        lower_to_real = {str(c).lower(): str(c) for c in df.columns}
        rename = {}
        mapping_used = {}

        # Step 1 — hardcoded aliases (fast path)
        for canonical, candidates in field_map.items():
            for cand in candidates:
                if cand.lower() in lower_to_real:
                    real = lower_to_real[cand.lower()]
                    rename[real] = canonical
                    mapping_used[canonical] = real
                    break

        # Step 2 — LLM fallback for unmapped source columns
        if target_entity and hasattr(self, "llm_mapper"):
            mapped_source_cols = set(rename.keys())
            unmapped_canonical = set(field_map.keys()) - set(mapping_used.keys())
            for col in df.columns:
                if str(col) in mapped_source_cols:
                    continue
                # Sample a few non-null values
                samples = [v for v in df[col].dropna().head(5).tolist()]
                if not samples:
                    continue
                suggestion = self.llm_mapper.suggest(target_entity, str(col), samples)
                if suggestion and suggestion["master_field"] in unmapped_canonical:
                    # Only use suggestions with >=0.7 confidence
                    if suggestion.get("confidence", 0) >= 0.7:
                        rename[str(col)] = suggestion["master_field"]
                        mapping_used[suggestion["master_field"]] = str(col)
                        unmapped_canonical.discard(suggestion["master_field"])

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
        # Record this dataset's source_system as first contributor to this PK
        try:
            ss_row = con.execute("SELECT source_system FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
            source_system = ss_row[0] if ss_row and ss_row[0] else dataset_id
            con.execute(
                "INSERT OR IGNORE INTO master_record_sources "
                "(master_table, pk_value, dataset_id, first_seen_at, source_system) VALUES (?, ?, ?, ?, ?)",
                (table, str(record[pk_column]), dataset_id, now, source_system),
            )
        except Exception:
            pass
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
        field_changes: list[dict[str, Any]] = []
        for k, v in new_record.items():
            if k not in existing_dict:
                continue
            current = existing_dict[k]
            if (current is None or str(current).strip() == "") and v is not None and str(v).strip() != "":
                updates[k] = v
                field_changes.append({
                    "column": k,
                    "old_value": current,
                    "new_value": v,
                })

        updates["last_updated"] = now
        # Look up the canonical source_system for this dataset (rest_api_employees,
        # file_customers, sqlite_legacy_customers, etc.) — re-runs of the SAME source
        # share the same source_system and therefore don\'t inflate the counter.
        ss_row = con.execute(
            "SELECT source_system FROM datasets WHERE dataset_id=?", (dataset_id,)
        ).fetchone()
        source_system = ss_row[0] if ss_row and ss_row[0] else dataset_id
        con.execute(
            "INSERT OR IGNORE INTO master_record_sources "
            "(master_table, pk_value, dataset_id, first_seen_at, source_system) VALUES (?, ?, ?, ?, ?)",
            (table, str(pk_value), dataset_id, now, source_system),
        )
        new_count = con.execute(
            "SELECT COUNT(DISTINCT source_system) FROM master_record_sources "
            "WHERE master_table=? AND pk_value=? AND source_system IS NOT NULL",
            (table, str(pk_value)),
        ).fetchone()[0]
        updates["merged_from_count"] = max(1, new_count)

        set_clause = ", ".join(f"{k}=?" for k in updates.keys())
        con.execute(
            f"UPDATE {table} SET {set_clause} WHERE {pk_column} = ?",
            (*updates.values(), pk_value),
        )
        enriched = [k for k in updates if k not in ("last_updated", "merged_from_count")]
        return {
            "fields_enriched": len(enriched),
            "enriched_columns": enriched,
            "field_changes":    field_changes,
        }

    def _table_count(self, entity: str) -> int:
        table, _, _ = MASTER_TABLES[entity]
        with sqlite3.connect(self.db_path, timeout=30) as con:
            return con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE archived_at IS NULL"
            ).fetchone()[0]
