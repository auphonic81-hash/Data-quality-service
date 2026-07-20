"""Staging layer (Informatica-style).

Each landing table maps to one staging table per master entity it contributes to.
Staging tables have:
  - Columns matching the master schema (canonical names, types)
  - Cleansed + standardized values (phones E.164, dates ISO, names Title Case)
  - source_system column (audit trail of where this row came from)
  - Validation status (pass/reject — invalid rows can be filtered before master load)

Flow: landing_<source>  →  stg_<entity>_<source>  →  master_<entity>
"""
from __future__ import annotations
import re
import sqlite3
import pandas as pd
from typing import Any

# Entity → canonical column schema (matches master_* table columns)
# Used to validate which columns must exist in staging
ENTITY_SCHEMAS: dict[str, list[str]] = {
    "Customers": ["customer_id", "full_name", "email", "phone", "country",
                  "city", "address", "status"],
    "Vendors":   ["vendor_id", "vendor_name", "contact_email", "contact_phone",
                  "country", "status"],
    "Products":  ["sku", "product_name", "category", "price", "msrp",
                  "supplier_id", "stock"],
    "Invoices":  ["invoice_no", "customer_id", "amount", "currency",
                  "invoice_date", "due_date", "status", "description"],
    "Employees": ["employee_id", "full_name", "email", "phone",
                  "department_code", "department_name", "country",
                  "hire_date", "status", "manager_id", "salary"],
}

# Common column-name variants → canonical name
COLUMN_ALIASES: dict[str, str] = {
    # IDs
    "cust_nr": "customer_id", "cust_id": "customer_id", "cust_key": "customer_id",
    "cust_no": "customer_id", "id": "customer_id", "customer": "customer_id",
    "emp_no": "employee_id", "emp_id": "employee_id",
    "vendor_nr": "vendor_id", "vendor_code": "vendor_id", "supplier_id": "vendor_id",
    "lead_id": "customer_id",
    "bill_no": "invoice_no", "invoice_id": "invoice_no", "order_number": "invoice_no",
    "order_id": "invoice_no",
    "prod_code": "sku", "product_id": "sku", "product_sku": "sku",
    "product_code": "sku",
    # Names
    "full_nm": "full_name", "name": "full_name", "contact_nm": "full_name",
    "emp_full_name": "full_name", "company_nm": "vendor_name",
    "vendor_nm": "vendor_name", "product_nm": "product_name",
    "name1": "full_name", "customer_name": "full_name",
    "company_name": "vendor_name", "desc1": "product_name",
    "description": "product_name",
    # Email
    "work_email": "email", "email_addr": "email", "email_adr": "email",
    # Phone
    "mobile_nr": "phone", "phone_nr": "phone", "tel_nr": "phone",
    "contact_tel": "contact_phone",
    # Country / city / address
    "ctry": "country", "ctry_cd": "country", "country_code": "country",
    "land": "country", "customer_country": "country", "shipping_country": "country",
    "stadt": "city", "customer_city": "city", "shipping_city": "city",
    "shipping_address": "address", "addr": "address",
    # Department
    "dept_cd": "department_code", "dept": "department_code",
    "dept_nm": "department_name",
    # Other
    "hire_dt": "hire_date", "annual_sal": "salary", "salary_usd": "salary",
    "status_cd": "status", "stat": "status", "order_status": "status",
    "payment_status": "status",
    "mgr_id": "manager_id",
    "amount": "amount", "amt": "amount", "opp_amt": "amount", "total": "amount",
    "unit_price": "price",
    "crt_dt": "invoice_date", "order_dt": "invoice_date", "order_date": "invoice_date",
    "due_dt": "due_date", "crdat": "invoice_date",
    "category": "category", "cat": "category", "product_category": "category",
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names so they line up with master schema."""
    rename_map = {}
    for col in df.columns:
        canonical = COLUMN_ALIASES.get(col.lower().strip())
        if canonical and canonical != col:
            rename_map[col] = canonical
        # Lowercase any column with no mapping (so case differences don't break joins)
        elif col != col.lower():
            rename_map[col] = col.lower()
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _infer_entity_from_landing(landing_name: str) -> str | None:
    """Heuristic: which master entity does this landing table belong to?"""
    n = landing_name.lower()
    if any(k in n for k in ("customer", "client", "contact")) and "audit" not in n:
        return "Customers"
    if "vendor" in n or "supplier" in n:
        return "Vendors"
    if "product" in n or "sku" in n:
        return "Products"
    if any(k in n for k in ("invoice", "bill", "order")) and "shipment" not in n:
        return "Invoices"
    if "employee" in n or "staff" in n:
        return "Employees"
    if "customer_audit" in n:
        return "Customers"
    return None


def build_staging_table(db_path: str, landing_name: str) -> dict:
    """Build a stg_<entity>_<source> table from a landing table.

    Cleansing happens during ingestion already (phones to E.164, names Title Case).
    Staging just extracts the entity-relevant columns, canonicalizes names,
    and adds source_system + validation status.
    """
    entity = _infer_entity_from_landing(landing_name)
    if not entity:
        return {"landing": landing_name, "status": "skipped",
                "reason": "no matching master entity"}

    con = sqlite3.connect(db_path, timeout=30)
    try:
        df = pd.read_sql(f"SELECT * FROM {landing_name}", con)
    finally:
        con.close()

    # Strip provenance cols + canonicalize column names
    df = df[[c for c in df.columns if not c.startswith("_landing_")]].copy()
    df = _canonicalize_columns(df)

    # Keep only columns the entity schema cares about (drop extras)
    schema_cols = ENTITY_SCHEMAS[entity]
    keep = [c for c in df.columns if c in schema_cols]
    if not keep:
        return {"landing": landing_name, "status": "skipped",
                "reason": "no entity-schema columns found after canonicalization"}
    stg_df = df[keep].copy()

    # Add source_system column (lookup from datasets table via landing name)
    source_system = landing_name.replace("landing_", "", 1)
    stg_df["_source_system"] = source_system
    stg_df["_staged_at"] = pd.Timestamp.utcnow().isoformat()

    # Validation: rows missing the entity's primary key get marked rejected
    pk_map = {
        "Customers":     "customer_id",
        "Vendors":       "vendor_id",
        "Products":      "sku",
        "Invoices":      "invoice_no",
        "Employees":     "employee_id",
        "Accounts":      "acct_id",
        "Contracts":     "contract_id",
        "Departments":   "department_code",
        "Inventory":     "inv_item_id",
        "Locations":     "loc_id",
        "Opportunities": "opp_id",
        "Payments":      "pmt_id",
        "Shipments":     "ship_id",
        "SalesOrders":   "order_id",
    }
    pk = pk_map.get(entity)
    if pk in stg_df.columns:
        stg_df["_validation"] = stg_df[pk].apply(
            lambda v: "ok" if v is not None and str(v).strip() else "rejected_null_pk"
        )
    else:
        stg_df["_validation"] = "rejected_no_pk_column"

    # Write to stg_<entity>_<source>
    stg_table = f"stg_{entity.lower()}__{source_system}"
    con = sqlite3.connect(db_path, timeout=30)
    try:
        stg_df.to_sql(stg_table, con, if_exists="replace", index=False)
        con.commit()
    finally:
        con.close()

    return {
        "landing": landing_name,
        "entity": entity,
        "staging_table": stg_table,
        "rows": int(len(stg_df)),
        "ok_rows": int((stg_df["_validation"] == "ok").sum()),
        "rejected_rows": int((stg_df["_validation"] != "ok").sum()),
        "columns_mapped": keep,
        "status": "ok",
    }


def build_all_staging_tables(db_path: str) -> dict:
    """Walk every landing table, produce its staging counterpart."""
    con = sqlite3.connect(db_path, timeout=30)
    try:
        landing_tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'landing_%' ORDER BY name"
        ).fetchall()]
    finally:
        con.close()

    results = []
    for lt in landing_tables:
        results.append(build_staging_table(db_path, lt))

    ok_count = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    return {
        "landing_tables_scanned": len(landing_tables),
        "staging_tables_created": ok_count,
        "skipped": skipped,
        "details": results,
    }


def list_staging_tables(db_path: str) -> list[dict]:
    con = sqlite3.connect(db_path, timeout=30)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'stg_%' ORDER BY name"
        ).fetchall()]
        out = []
        for n in names:
            cnt = con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]
            ok_cnt = con.execute(f"SELECT COUNT(*) FROM {n} WHERE _validation='ok'").fetchone()[0]
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({n})").fetchall()]
            out.append({
                "table": n, "rows": int(cnt), "ok_rows": int(ok_cnt),
                "columns": cols, "col_count": len(cols),
            })
        return out
    finally:
        con.close()
