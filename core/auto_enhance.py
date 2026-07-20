"""Auto-enhancement on ingestion — Informatica-style architecture.

When a new source arrives:
1. Cleansing has already happened (in ingestion.py)
2. Landing table is written by service.py
3. THIS module: AutoNormalize splits the landing data via DFD discovery
4. Each normalized child gets routed to its target master entity
   - If matches an existing master → MERGE into that master
   - If no match → CREATE a new master entity from the child

Result: master tables are inherently normalized. No post-cleanup needed.
"""
from __future__ import annotations
import re
import sqlite3
import pandas as pd
from typing import Any


def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Run staging's FULL canonicalization (column rename + status codes + geo lookup).
    Single source of truth: same cleansing whether routed via auto_enhance or staging.
    """
    from .staging import _canonicalize_columns
    return _canonicalize_columns(df.copy())


# Patterns mapping PK column name → canonical master entity
PK_TO_ENTITY = {
    "customer_id": "Customers",
    "cust_id":     "Customers",
    "cust_no":     "Customers",
    "vendor_id":   "Vendors",
    "supplier_id": "Vendors",
    "vendor_code": "Vendors",
    "sku":         "Products",
    "product_id":  "Products",
    "prod_code":   "Products",
    "employee_id": "Employees",
    "emp_id":      "Employees",
    "emp_no":      "Employees",
    "invoice_no":  "Invoices",
    "invoice_id":  "Invoices",
    "bill_no":     "Invoices",
    "order_id":    "Invoices",
    "order_number":"Invoices",
    "dept_cd":     "Departments",
    "dept_id":     "Departments",
    "department_code": "Departments",
    "acct_id":     "Accounts",
    "account_id":  "Accounts",
    "contract_id": "Contracts",
    "loc_id":      "Locations",
    "location_id": "Locations",
    "opp_id":      "Opportunities",
    "pmt_id":      "Payments",
    "payment_id":  "Payments",
    "ship_id":     "Shipments",
    "inv_item_id": "Inventory",
    "warehouse_id":"Warehouses",
    "manager_id":  "Managers",
}


def _entity_for_pk(pk_col: str, fallback_stem: str = "") -> str:
    """Map a PK column name to an entity. Fallback: PascalCase the source stem."""
    canon = pk_col.lower().strip()
    if canon in PK_TO_ENTITY:
        return PK_TO_ENTITY[canon]
    # Derive from PK if it ends with _id/_no/_key — e.g. warehouse_id → Warehouses
    for suffix in ("_id", "_no", "_key", "_code", "_nr"):
        if canon.endswith(suffix):
            stem = canon[:-len(suffix)]
            if stem:
                return stem.capitalize() + "s"
    # Final fallback: source-table stem (strip api_/sqlite_/file_/rest_api_ prefixes)
    if fallback_stem:
        stem = fallback_stem
        for prefix in ("rest_api_", "api_", "sqlite_", "file_", "pdf_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        parts = re.split(r"[_\s]+", stem)
        return "".join(p.capitalize() for p in parts if p)
    return "Unknown"


def _ensure_master_table(con, master_tbl: str, df: pd.DataFrame, source_system: str,
                          landing_table_name: str) -> dict:
    """Create or merge into a master_<entity> table.

    - If table doesn't exist: create from this child + add audit cols
    - If table exists: INSERT new PK values, skip existing
    """
    exists = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (master_tbl,)
    ).fetchone()

    df = df.copy()
    df["_source_system"] = source_system
    df["_promoted_from"] = landing_table_name
    df["_created_at"]    = pd.Timestamp.utcnow().isoformat()

    if not exists:
        df.to_sql(master_tbl, con, if_exists="replace", index=False)
        return {"action": "created", "rows_inserted": int(len(df))}
    else:
        # Detect master's PK (first non-underscore column we wrote)
        master_cols = [c[1] for c in con.execute(f"PRAGMA table_info({master_tbl})").fetchall()]
        pk_col = next((c for c in df.columns if not c.startswith("_") and c in master_cols), None)
        if not pk_col:
            return {"action": "skipped", "reason": "no common PK column"}
        existing_pks = set(r[0] for r in con.execute(f"SELECT {pk_col} FROM {master_tbl}").fetchall())
        new_rows = df[~df[pk_col].isin(existing_pks)]
        if new_rows.empty:
            return {"action": "merged", "rows_inserted": 0, "rows_existing": int(len(df))}
        # Align columns (insert only cols that exist in master, fill missing with NULL)
        for c in new_rows.columns:
            if c not in master_cols:
                # Add missing column to master
                try:
                    con.execute(f"ALTER TABLE {master_tbl} ADD COLUMN {c} TEXT")
                    master_cols.append(c)
                except Exception:
                    pass
        new_rows[[c for c in new_rows.columns if c in master_cols]].to_sql(
            master_tbl, con, if_exists="append", index=False
        )
        return {"action": "merged", "rows_inserted": int(len(new_rows)),
                "rows_existing": int(len(df) - len(new_rows))}


def auto_enhance_after_ingestion(db_path: str, landing_table_name: str) -> dict:
    # Decide whether to auto-promote a new entity from this landing table.
    # Old logic: filename-keyword match. That missed random garbage files
    # like landing_file_random_garbage which then got auto-promoted to
    # master_randomgarbage with nonsense columns.
    # New logic: defer to the entity classifier. If it identifies a known
    # entity, master_repo handles it (we skip here). If it returns None
    # (unrecognized), DO NOT auto-promote — the file shouldn't become a
    # master entity; the disposition layer will aside the source.
    try:
        import sqlite3 as _sql_skip
        import pandas as _pd_skip
        from .entity_classifier import classify_entity as _classify_skip
        _con_skip = _sql_skip.connect(db_path, timeout=30)
        try:
            _sample = _pd_skip.read_sql_query(
                f"SELECT * FROM {landing_table_name} LIMIT 200", _con_skip
            )
        finally:
            _con_skip.close()
        if not _sample.empty:
            _classified, _conf, _dbg = _classify_skip(_sample, landing_table_name)
            if _classified is not None:
                return {
                    "skipped": True,
                    "reason": f"classified as {_classified} (conf {_conf}) - master_repo handles it",
                    "normalized_children": [],
                }
            else:
                return {
                    "skipped": True,
                    "reason": "classifier returned no confident match - not auto-promoting; disposition layer will aside the source file",
                    "normalized_children": [],
                }
    except Exception as _e_skip:
        # Classifier failed for some reason - fall back to old keyword check
        print(f"[auto_enhance] classifier check failed, falling back to keywords: {_e_skip}")
        _known = ("customer", "vendor", "supplier", "product", "invoice", "bill",
                  "order", "employee", "staff")
        if any(k in landing_table_name.lower() for k in _known):
            return {"skipped": True, "reason": "known entity routed via master_repo (fallback)", "normalized_children": []}

    """Main entry point — called from service.py after a landing table is written.

    Pipeline:
      1. Load landing data
      2. Canonicalize columns
      3. AutoNormalize → list of normalized child tables (the proper 3NF split)
      4. For each child: route to its master entity (existing or new)
      5. Result: every master table is naturally normalized
    """
    report = {
        "landing_table":     landing_table_name,
        "children_produced": 0,
        "routed":            [],
    }

    con = sqlite3.connect(db_path, timeout=30)
    try:
        df_raw = pd.read_sql(f"SELECT * FROM {landing_table_name}", con)
    finally:
        con.close()

    if df_raw.empty:
        return {**report, "status": "skipped_empty"}

    # Strip provenance + canonicalize column names
    df = df_raw[[c for c in df_raw.columns if not c.startswith("_landing_")]].copy()
    df_canon = _canonicalize(df)

    source_system = landing_table_name.replace("landing_", "", 1)
    source_stem = source_system

    # Step 1: AutoNormalize (only on tables wide enough + with rows enough)
    children = []
    if 4 <= len(df_canon.columns) <= 15 and len(df_canon) >= 5:
        try:
            import autonormalize as _an
            children = _an.auto_normalize(df_canon)
        except Exception as exc:
            report["normalize_error"] = str(exc)
            children = []

    if not children:
        # Single-table promotion: no FDs found, whole landing → one master
        entity = _entity_for_pk(df_canon.columns[0], fallback_stem=source_stem) if len(df_canon.columns) else "Unknown"
        master_tbl = f"master_{entity.lower()}"
        try:
            con = sqlite3.connect(db_path, timeout=30)
            result = _ensure_master_table(con, master_tbl, df_canon, source_system, landing_table_name)
            con.commit()
            con.close()
            report["routed"].append({"entity": entity, "master_table": master_tbl, **result})
        except Exception as exc:
            report["error"] = str(exc)
        return {**report, "status": "ok", "children_produced": 0}

    # Step 2: Route each normalized child to its master entity
    report["children_produced"] = len(children)
    con = sqlite3.connect(db_path, timeout=30)
    try:
        for child in children:
            if child.empty or len(child.columns) == 0:
                continue
            pk_col = str(child.columns[0])
            entity = _entity_for_pk(pk_col, fallback_stem=source_stem)
            master_tbl = f"master_{entity.lower()}"
            try:
                result = _ensure_master_table(con, master_tbl, child, source_system, landing_table_name)
                report["routed"].append({
                    "entity":         entity,
                    "master_table":   master_tbl,
                    "pk_column":      pk_col,
                    "columns":        list(child.columns),
                    "child_rows":     int(len(child)),
                    **result,
                })
            except Exception as exc:
                report["routed"].append({"entity": entity, "error": str(exc)})
        con.commit()
    finally:
        con.close()

    return {**report, "status": "ok"}
