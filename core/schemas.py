"""Pandera schemas — production data quality enforcement.

Two-tier system:
1. CURATED schemas (this file) — hand-tuned for the 13 known entities with
   PK regex, FK checks, entity-scoped status enums. Highest authority.

2. DYNAMIC schemas (auto-generated) — for anything AutoNormalize promotes.
   Inferred from the dataframe at ingestion, persisted in SQLite, refined
   over time. Three lifecycle states:
     - learning:    first N rows, observe patterns, log anomalies, accept
     - established: threshold reached, schema locked, violations rejected
     - curated:     steward promoted/edited, treated as authoritative

Rows that fail validation go to master_rejections_v2 with a reason string.
"""
from __future__ import annotations
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check


# ===========================================================================
# Reusable atomic checks
# ===========================================================================
def _phone_check_fn(s):
    if s is None or not isinstance(s, str):
        return True
    digits = "".join(c for c in s if c.isdigit())
    return 8 <= len(digits) <= 15

_phone_e164  = Check(lambda series: series.astype(str).apply(_phone_check_fn),
                     error="phone must have 8-15 digits")
_email_basic = Check.str_matches(r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                                 error="email must be valid format")
_non_empty   = Check(lambda s: s.astype(str).str.strip().str.len() > 0,
                     error="value cannot be empty")

# PK format checks
_pk_customer    = Check.str_matches(r"^C-\d{4,6}$",       error="customer_id must match C-NNNNN")
_pk_vendor      = Check.str_matches(r"^V-\d{4,6}$",       error="vendor_id must match V-NNNN")
_pk_employee    = Check.str_matches(r"^E-\d{4,6}$",       error="employee_id must match E-NNNNN")
_pk_account     = Check.str_matches(r"^A-\d{4,6}$",       error="acct_id must match A-NNNN")
_pk_product     = Check.str_matches(r"^(SKU-[A-Z0-9-]+|P-\d+)$",  error="sku must match SKU-XXXX or P-NNNN")
_pk_invoice     = Check.str_matches(r"^(INV|BILL)-(\d{4}-)?\d{4,6}$",
                                    error="invoice_no must match INV-NNNNNN or INV-YYYY-NNNNN")
_pk_contract    = Check.str_matches(r"^CT-\d{4,6}$",      error="contract_id must match CT-NNNNN")
_pk_department  = Check.str_matches(r"^D-\d{3,5}$",       error="department_code must match D-NNN")
_pk_location    = Check.str_matches(r"^LOC-\d{4,6}$",     error="loc_id must match LOC-NNNNN")
_pk_opportunity = Check.str_matches(r"^OPP-\d{4,6}$",     error="opp_id must match OPP-NNNNNN")
_pk_payment     = Check.str_matches(r"^PMT-\d{4,6}$",     error="pmt_id must match PMT-NNNNNN")
_pk_shipment    = Check.str_matches(r"^SHP-\d{4,6}$",     error="ship_id must match SHP-NNNNNN")
_pk_inventory   = Check.str_matches(r"^INVT-\d{4,6}$",    error="inv_item_id must match INVT-NNNNNN")
_pk_salesorder  = Check.str_matches(r"^(SO|ORD|O)-\d{4,6}$",  error="order_id must match SO-NNNNNN")

# Entity-scoped status enums
_status_party = Check.isin(
    ["Active", "Inactive", "Terminated", "On Leave", "Pending", "Suspended", "Unknown"],
    error="status must be Active/Inactive/Terminated/On Leave/Pending/Suspended/Unknown"
)
_status_invoice = Check.isin(
    ["Paid", "Unpaid", "Open", "Pending", "Overdue", "Cancelled", "Refunded", "Draft", "Unknown"],
    error="invoice status must be Paid/Unpaid/Open/Pending/Overdue/Cancelled/Refunded/Draft/Unknown"
)
_status_shipment = Check.isin(
    ["Shipped", "Delivered", "In Transit", "Processing", "Returned", "Lost", "Cancelled", "Unknown"],
    error="shipment status must be Shipped/Delivered/In Transit/Processing/Returned/Lost/Cancelled/Unknown"
)
_status_opportunity = Check.isin(
    ["Open", "Won", "Lost", "In Progress", "Qualified", "Proposal", "Negotiation",
     "Closed", "Closed-Won", "Closed-Lost", "Prospect", "Unknown"],
    error="opportunity status must be Open/Won/Lost/Closed/Closed-Won/Closed-Lost/Prospect/Qualified/Proposal/Negotiation/In Progress/Unknown"
)
_status_contract = Check.isin(
    ["Active", "Expired", "Terminated", "Pending", "Cancelled", "Renewed", "Draft", "Unknown"],
    error="contract status must be Active/Expired/Terminated/Pending/Cancelled/Renewed/Draft/Unknown"
)
_status_salesorder = Check.isin(
    ["Open", "Shipped", "Delivered", "Cancelled", "Returned", "Pending", "Processing", "Unknown"],
    error="order status must be Open/Shipped/Delivered/Cancelled/Returned/Pending/Processing/Unknown"
)


# ===========================================================================
# CURATED SCHEMAS — 13 known entities
# ===========================================================================

CUSTOMER_SCHEMA = DataFrameSchema(
    {
        "customer_id": Column(str, checks=[_non_empty, _pk_customer], required=True),
        "full_name":   Column(str, checks=[_non_empty], required=True),
        "email":       Column(str, checks=[_email_basic], required=False, nullable=True),
        "phone":       Column(str, checks=[_phone_e164], required=False, nullable=True),
        "status":      Column(str, checks=[_status_party], required=False, nullable=True),
        "country":     Column(str, required=False, nullable=True),
    },
    strict=False, coerce=True,
)

def _forbid_full_name(df):
    if "full_name" not in df.columns:
        return True
    return df["full_name"].isna() | (df["full_name"].astype(str).str.strip() == "")

VENDOR_SCHEMA = DataFrameSchema(
    {
        "vendor_id":     Column(str, checks=[_non_empty, _pk_vendor], required=True),
        "vendor_name":   Column(str, checks=[_non_empty], required=True),
        "contact_phone": Column(str, checks=[_phone_e164], required=False, nullable=True),
        "status":        Column(str, checks=[_status_party], required=False, nullable=True),
        "country":       Column(str, required=False, nullable=True),
    },
    strict=False, coerce=True,
    checks=[Check(_forbid_full_name, error="full_name not allowed on Vendor entity")],
)

PRODUCT_SCHEMA = DataFrameSchema(
    {
        "sku":          Column(str, checks=[_non_empty, _pk_product], required=True),
        "product_name": Column(str, checks=[_non_empty], required=True),
        "vendor_id":    Column(str, checks=[_pk_vendor], required=False, nullable=True),
        "price":        Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "msrp":         Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "category":     Column(str, required=False, nullable=True),
    },
    strict=False, coerce=True,
)

_pk_party = Check.str_matches(
    r"^(C-\d{4,6}|V-\d{4,6})$",
    error="customer_id must match C-NNNNN (customer) or V-NNNN (vendor)"
)

INVOICE_SCHEMA = DataFrameSchema(
    {
        "invoice_no":   Column(str, checks=[_non_empty, _pk_invoice], required=True),
        # customer_id accepts either C-NNNNN (customer) or V-NNNN (vendor)
        # because B2B bills can be addressed to either party in the master.
        "customer_id":  Column(str, checks=[_pk_party], required=False, nullable=True),
        "amount":       Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "status":       Column(str, checks=[_status_invoice], required=False, nullable=True),
    },
    strict=False, coerce=True,
)

EMPLOYEE_SCHEMA = DataFrameSchema(
    {
        "employee_id": Column(str, checks=[_non_empty, _pk_employee], required=True),
        "full_name":   Column(str, checks=[_non_empty], required=True),
        "email":       Column(str, checks=[_email_basic], required=False, nullable=True),
        "phone":       Column(str, checks=[_phone_e164], required=False, nullable=True),
        "status":      Column(str, checks=[_status_party], required=False, nullable=True),
        "salary":      Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "country":     Column(str, required=False, nullable=True),
        "city":        Column(str, required=False, nullable=True),
        "department_code": Column(str, checks=[_pk_department], required=False, nullable=True),
    },
    strict=False, coerce=True,
)

ACCOUNT_SCHEMA = DataFrameSchema(
    {
        "acct_id":         Column(str, checks=[_non_empty, _pk_account], required=True),
        "account_name":    Column(str, checks=[_non_empty], required=False, nullable=True),
        "annual_revenue":  Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "employee_count":  Column(int, checks=[Check.ge(0)], required=False, nullable=True),
        "industry":        Column(str, required=False, nullable=True),
        "country":         Column(str, required=False, nullable=True),
        "status":          Column(str, checks=[_status_party], required=False, nullable=True),
    },
    strict=False, coerce=True,
)

CONTRACT_SCHEMA = DataFrameSchema(
    {
        "contract_id":    Column(str, checks=[_non_empty, _pk_contract], required=True),
        "acct_id":        Column(str, checks=[_pk_account], required=False, nullable=True),
        "contract_value": Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "start_dt":       Column(str, required=False, nullable=True),
        "end_dt":         Column(str, required=False, nullable=True),
        "status":         Column(str, checks=[_status_contract], required=False, nullable=True),
    },
    strict=False, coerce=True,
)

DEPARTMENT_SCHEMA = DataFrameSchema(
    {
        "department_code": Column(str, checks=[_non_empty, _pk_department], required=True),
        "dept_name":       Column(str, checks=[_non_empty], required=True),
        "cost_center":     Column(str, required=False, nullable=True),
        "manager_emp_id":  Column(str, checks=[_pk_employee], required=False, nullable=True),
    },
    strict=False, coerce=True,
)

LOCATION_SCHEMA = DataFrameSchema(
    {
        "loc_id":  Column(str, checks=[_non_empty, _pk_location], required=True),
        "city":    Column(str, checks=[_non_empty], required=False, nullable=True),
        "country": Column(str, required=False, nullable=True),
        "acct_id": Column(str, checks=[_pk_account], required=False, nullable=True),
    },
    strict=False, coerce=True,
)

OPPORTUNITY_SCHEMA = DataFrameSchema(
    {
        "opp_id":       Column(str, checks=[_non_empty, _pk_opportunity], required=True),
        "opp_name":     Column(str, checks=[_non_empty], required=False, nullable=True),
        "acct_id":      Column(str, checks=[_pk_account], required=False, nullable=True),
        "owner_emp_id": Column(str, checks=[_pk_employee], required=False, nullable=True),
        "amount":       Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "status":       Column(str, checks=[_status_opportunity], required=False, nullable=True),
        "close_dt":     Column(str, required=False, nullable=True),
    },
    strict=False, coerce=True,
)

PAYMENT_SCHEMA = DataFrameSchema(
    {
        "pmt_id":       Column(str, checks=[_non_empty, _pk_payment], required=True),
        "pmt_amt":      Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "reference_no": Column(str, required=False, nullable=True),
        "pmt_dt":       Column(str, required=False, nullable=True),
        "invoice_no":   Column(str, checks=[_pk_invoice], required=False, nullable=True),
        "customer_id":  Column(str, checks=[_pk_customer], required=False, nullable=True),
        "method":       Column(str, required=False, nullable=True),
    },
    strict=False, coerce=True,
)

SHIPMENT_SCHEMA = DataFrameSchema(
    {
        "ship_id":     Column(str, checks=[_non_empty, _pk_shipment], required=True),
        "invoice_no":  Column(str, required=False, nullable=True),
        "tracking_no": Column(str, required=False, nullable=True),
        "shipped_dt":  Column(str, required=False, nullable=True),
        "status":      Column(str, checks=[_status_shipment], required=False, nullable=True),
    },
    strict=False, coerce=True,
)

INVENTORY_SCHEMA = DataFrameSchema(
    {
        "inv_item_id":   Column(str, checks=[_non_empty, _pk_inventory], required=True),
        "sku":           Column(str, checks=[_pk_product], required=False, nullable=True),
        "loc_id":        Column(str, checks=[_pk_location], required=False, nullable=True),
        "quantity":      Column(int, checks=[Check.ge(0)], required=False, nullable=True),
        "unit_cost":     Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "last_count_dt": Column(str, required=False, nullable=True),
    },
    strict=False, coerce=True,
)

SALESORDER_SCHEMA = DataFrameSchema(
    {
        "order_id":     Column(str, checks=[_non_empty, _pk_salesorder], required=True),
        "customer_id":  Column(str, checks=[_pk_customer], required=False, nullable=True),
        "product_id":   Column(str, required=False, nullable=True),
        "amount":       Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "quantity":     Column(int, checks=[Check.ge(0)], required=False, nullable=True),
        "total_amt":    Column(float, checks=[Check.ge(0)], required=False, nullable=True),
        "order_date":   Column(str, required=False, nullable=True),
        "status":       Column(str, checks=[_status_salesorder], required=False, nullable=True),
    },
    strict=False, coerce=True,
)

ENTITY_SCHEMAS = {
    "Customers":     CUSTOMER_SCHEMA,
    "Vendors":       VENDOR_SCHEMA,
    "Products":      PRODUCT_SCHEMA,
    "Invoices":      INVOICE_SCHEMA,
    "Employees":     EMPLOYEE_SCHEMA,
    "Accounts":      ACCOUNT_SCHEMA,
    "Contracts":     CONTRACT_SCHEMA,
    "Departments":   DEPARTMENT_SCHEMA,
    "Locations":     LOCATION_SCHEMA,
    "Opportunities": OPPORTUNITY_SCHEMA,
    "Payments":      PAYMENT_SCHEMA,
    "Shipments":     SHIPMENT_SCHEMA,
    "Inventory":     INVENTORY_SCHEMA,
    "SalesOrders":   SALESORDER_SCHEMA,
}


# ===========================================================================
# DYNAMIC SCHEMAS — auto-generated for unknown entities
# ===========================================================================

_CATALOG_PATH = Path(__file__).resolve().parent.parent / "reports" / "catalog.sqlite3"
_ESTABLISHMENT_THRESHOLD = 100


def _get_conn():
    conn = sqlite3.connect(str(_CATALOG_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def _ensure_table():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_schemas (
                entity TEXT PRIMARY KEY,
                schema_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'learning',
                row_count_observed INTEGER NOT NULL DEFAULT 0,
                threshold INTEGER NOT NULL DEFAULT 100,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_violation_at TEXT
            )
        """)


def _infer_id_pattern(values: pd.Series) -> Optional[str]:
    sample = values.head(50).tolist()
    if not sample:
        return None
    candidates = [
        r"^[A-Z]+-\d+$",
        r"^[A-Z]+-\d+-\d+$",
        r"^[A-Z]+-[A-Z0-9-]+$",
        r"^\d+$",
        r"^[A-Z0-9]+$",
    ]
    for pattern in candidates:
        if all(re.match(pattern, str(v)) for v in sample):
            return pattern
    return None


def _infer_column_metadata(series: pd.Series, col_name: str) -> dict:
    col_lower = col_name.lower()
    meta = {"dtype": "str", "nullable": bool(series.isna().any()), "checks": []}

    if pd.api.types.is_integer_dtype(series):
        meta["dtype"] = "int"
        if (series.dropna() >= 0).all():
            meta["checks"].append({"type": "ge", "value": 0})
    elif pd.api.types.is_float_dtype(series):
        meta["dtype"] = "float"
        if (series.dropna() >= 0).all():
            meta["checks"].append({"type": "ge", "value": 0})

    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return meta

    if "email" in col_lower:
        meta["checks"].append({"type": "email"})
    elif "phone" in col_lower or "mobile" in col_lower or "tel" in col_lower:
        meta["checks"].append({"type": "phone"})
    elif "status" in col_lower or col_lower.endswith("_state"):
        observed = sorted(set(non_null.tolist()))
        if 1 < len(observed) <= 20:
            meta["checks"].append({"type": "isin", "values": observed})
    elif col_lower.endswith("_id") or col_lower.endswith("_code") or col_lower.endswith("_no"):
        pattern = _infer_id_pattern(non_null)
        if pattern:
            meta["checks"].append({"type": "regex", "value": pattern})
        if not meta["nullable"]:
            meta["checks"].append({"type": "non_empty"})

    return meta


def _build_schema_from_metadata(metadata: dict, strict: bool) -> DataFrameSchema:
    columns = {}
    for col_name, meta in metadata["columns"].items():
        dtype = {"str": str, "int": int, "float": float}[meta["dtype"]]
        checks = []
        if strict:
            for c in meta["checks"]:
                if c["type"] == "email":
                    checks.append(_email_basic)
                elif c["type"] == "phone":
                    checks.append(_phone_e164)
                elif c["type"] == "non_empty":
                    checks.append(_non_empty)
                elif c["type"] == "isin":
                    checks.append(Check.isin(
                        c["values"],
                        error=f"{col_name} must be one of {c['values']}"
                    ))
                elif c["type"] == "regex":
                    checks.append(Check.str_matches(
                        c["value"],
                        error=f"{col_name} must match {c['value']}"
                    ))
                elif c["type"] == "ge":
                    checks.append(Check.ge(c["value"]))
        columns[col_name] = Column(
            dtype, checks=checks, required=False, nullable=meta["nullable"],
        )
    return DataFrameSchema(columns, strict=False, coerce=True)


def _merge_metadata(old: dict, new: dict) -> dict:
    merged = {"columns": dict(old.get("columns", {}))}
    for col, new_meta in new["columns"].items():
        if col not in merged["columns"]:
            merged["columns"][col] = new_meta
            continue
        old_meta = merged["columns"][col]
        old_meta["nullable"] = old_meta["nullable"] or new_meta["nullable"]
        for new_check in new_meta["checks"]:
            if new_check["type"] == "isin":
                existing_isin = next(
                    (c for c in old_meta["checks"] if c["type"] == "isin"), None
                )
                if existing_isin:
                    existing_isin["values"] = sorted(
                        set(existing_isin["values"]) | set(new_check["values"])
                    )
                else:
                    old_meta["checks"].append(new_check)
    return merged


def _load_dynamic_schema(entity: str):
    try:
        _ensure_table()
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT schema_json, status, row_count_observed FROM dynamic_schemas WHERE entity = ?",
                (entity,)
            ).fetchone()
        if not row:
            return None
        metadata = json.loads(row[0])
        status = row[1]
        observed = row[2]
        strict = status in ("established", "curated")
        schema = _build_schema_from_metadata(metadata, strict=strict)
        return schema, status, observed
    except Exception as e:
        print(f"[dynamic_schemas] load error for {entity}: {e}")
        return None


def _persist_dynamic_schema(entity: str, df: pd.DataFrame):
    try:
        _ensure_table()
        now = datetime.utcnow().isoformat()
        metadata = {
            "columns": {col: _infer_column_metadata(df[col], col) for col in df.columns}
        }
        with _get_conn() as conn:
            existing = conn.execute(
                "SELECT row_count_observed, status, schema_json FROM dynamic_schemas WHERE entity = ?",
                (entity,)
            ).fetchone()
            if existing:
                old_count, old_status, old_json = existing
                new_count = old_count + len(df)
                if old_status == "curated":
                    new_status = "curated"
                elif new_count >= _ESTABLISHMENT_THRESHOLD:
                    new_status = "established"
                else:
                    new_status = "learning"
                metadata = _merge_metadata(json.loads(old_json), metadata)
                conn.execute("""
                    UPDATE dynamic_schemas
                    SET schema_json = ?, status = ?, row_count_observed = ?, updated_at = ?
                    WHERE entity = ?
                """, (json.dumps(metadata), new_status, new_count, now, entity))
            else:
                conn.execute("""
                    INSERT INTO dynamic_schemas
                    (entity, schema_json, status, row_count_observed, threshold, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (entity, json.dumps(metadata), "learning", len(df),
                      _ESTABLISHMENT_THRESHOLD, now, now))
    except Exception as e:
        print(f"[dynamic_schemas] persist error for {entity}: {e}")


# ===========================================================================
# Public entrypoint
# ===========================================================================

def validate(df, entity: str):
    """Validate df against the entity's schema.

    Returns (valid_df, rejected_df, error_log).

    Resolution order:
      1. Curated schema (this file) — authoritative
      2. Dynamic schema (SQLite) — auto-generated, may be learning or established
      3. Fresh inference — first time we've seen this entity, learn from df
    """
    schema = ENTITY_SCHEMAS.get(entity)
    if schema is not None:
        return _run_validation(schema, df)

    loaded = _load_dynamic_schema(entity)
    if loaded is not None:
        schema, status, _observed = loaded
        _persist_dynamic_schema(entity, df)
        if status == "learning":
            return df, df.iloc[0:0], []
        return _run_validation(schema, df)

    _persist_dynamic_schema(entity, df)
    return df, df.iloc[0:0], []


def _run_validation(schema: DataFrameSchema, df: pd.DataFrame):
    try:
        validated = schema.validate(df, lazy=True)
        return validated, df.iloc[0:0], []
    except pa.errors.SchemaErrors as exc:
        failure_cases = exc.failure_cases
        bad_indices = set(failure_cases["index"].dropna().astype(int).tolist())
        bad_mask = df.index.isin(bad_indices)
        rejected = df[bad_mask].copy()
        rejected["_rejection_reason"] = rejected.index.map(
            lambda i: "; ".join(
                f"{r['column']}: {r['check']}"
                for _, r in failure_cases[failure_cases["index"] == i].iterrows()
            )
        )
        valid = df[~bad_mask].copy()
        errors = failure_cases.to_dict("records")
        return valid, rejected, errors


# ===========================================================================
# Steward management API (for UI integration later)
# ===========================================================================

def list_dynamic_schemas():
    """Return all dynamic schemas with their lifecycle status."""
    try:
        _ensure_table()
        with _get_conn() as conn:
            rows = conn.execute("""
                SELECT entity, status, row_count_observed, threshold, created_at, updated_at
                FROM dynamic_schemas
                ORDER BY entity
            """).fetchall()
        return [
            {"entity": r[0], "status": r[1], "observed": r[2],
             "threshold": r[3], "created_at": r[4], "updated_at": r[5]}
            for r in rows
        ]
    except Exception as e:
        print(f"[dynamic_schemas] list error: {e}")
        return []


def promote_to_curated(entity: str) -> bool:
    """Steward action — lock a dynamic schema as curated.

    After this, the schema won't auto-update from new observations.
    """
    try:
        _ensure_table()
        with _get_conn() as conn:
            cur = conn.execute(
                "UPDATE dynamic_schemas SET status = 'curated', updated_at = ? WHERE entity = ?",
                (datetime.utcnow().isoformat(), entity)
            )
        return cur.rowcount > 0
    except Exception as e:
        print(f"[dynamic_schemas] promote error: {e}")
        return False