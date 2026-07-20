"""Landing layer — Informatica-style raw data persistence.

Every ingested source file becomes a `landing_<source>` table in the catalog DB.
This is Layer 1 of the 3-layer architecture (Landing → Normalized → Master).
"""
from __future__ import annotations
import sqlite3
import re
import pandas as pd
from pathlib import Path


def _safe_table_name(source_name: str) -> str:
    """Convert a source name into a safe SQL table name.
    'customers.csv' → 'landing_customers'
    'orders__Customers.csv' → 'landing_orders__customers'
    'api_employees' → 'landing_api_employees'
    'db_legacy_customers' → 'landing_legacy_customers'
    """
    name = (source_name or "unknown").strip().lower()
    # Strip extensions
    name = re.sub(r'\.(csv|xlsx?|json|tsv|pdf)$', '', name)
    # Strip pipeline prefixes (we'll re-add our own)
    if name.startswith('api_'):
        name = name[4:]
        prefix = 'api_'
    elif name.startswith('db_'):
        name = name[3:]
        prefix = 'sqlite_'
    else:
        prefix = 'file_'
    # Replace anything non-alphanumeric/underscore with _
    name = re.sub(r'[^a-z0-9_]+', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return f"landing_{prefix}{name}"


def persist_to_landing(db_path: str, source_name: str, df: pd.DataFrame, source_system: str) -> str:
    """Write a DataFrame to a landing_<source> table. Returns the table name created."""
    if df is None or df.empty:
        return ""
    # source_system hints at the type (pdf_, rest_api_, sqlite_, file_) — use it for the table prefix
    if source_system and source_system.startswith("pdf_"):
        # PDF extracted data
        name = source_system  # already "pdf_<stem>"
        table = "landing_" + name.replace("/", "_").replace(" ", "_")
    else:
        table = _safe_table_name(source_name)
    # Make a copy and add provenance columns
    persist_df = df.copy()
    persist_df['_landing_source_system'] = source_system
    persist_df['_landing_ingested_at'] = pd.Timestamp.utcnow().isoformat()
    # Coerce all columns to string to avoid SQLite type issues with mixed types
    for col in persist_df.columns:
        if persist_df[col].dtype == 'object' or 'date' in str(persist_df[col].dtype):
            persist_df[col] = persist_df[col].astype(str)
    con = sqlite3.connect(db_path, timeout=30)
    try:
        persist_df.to_sql(table, con, if_exists='replace', index=False)
        con.commit()
    finally:
        con.close()
    return table


def list_landing_tables(db_path: str) -> list[dict]:
    """Return all landing_* tables with row counts and column counts."""
    con = sqlite3.connect(db_path, timeout=30)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'landing_%' ORDER BY name"
        ).fetchall()]
        result = []
        for n in names:
            row_count = con.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0]
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({n})").fetchall()]
            # Exclude our internal provenance cols from the visible count
            visible_cols = [c for c in cols if not c.startswith('_landing_')]
            result.append({
                'table_name': n,
                'rows': row_count,
                'column_count': len(visible_cols),
                'columns': visible_cols,
            })
        return result
    finally:
        con.close()
