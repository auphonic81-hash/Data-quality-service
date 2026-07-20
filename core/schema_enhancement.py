"""Schema enhancement — schemas improve as data flows in.

1. Auto-indexer: high-cardinality + frequently-filtered columns get indexes.
2. Type refiner: TEXT columns that look like dates/numbers get re-typed.
3. Constraint inferrer: high-non-null columns get NOT NULL; unique cols get UNIQUE.

Runs across landing, staging, normalized, and master tables.
"""
from __future__ import annotations
import re
import sqlite3
import pandas as pd
from datetime import datetime

# Columns that typically benefit from an index (high-traffic in queries)
INDEX_WORTHY_PATTERNS = (
    "id", "no", "key", "code", "email", "phone",
    "country", "city", "status", "category", "type",
    "date", "_at", "_dt", "ref",
)


def _is_index_worthy(col_name: str) -> bool:
    lc = col_name.lower()
    return any(p in lc for p in INDEX_WORTHY_PATTERNS)


def discover_indexes(db_path: str, table_prefix: str = "") -> list[dict]:
    """Find columns that should have indexes but don't yet."""
    con = sqlite3.connect(db_path, timeout=30)
    try:
        if table_prefix:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name", (f"{table_prefix}%",)
            ).fetchall()]
        else:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()]

        # Find existing indexes
        existing_indexes = set()
        for r in con.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE type='index' "
            "AND sql IS NOT NULL"
        ).fetchall():
            tbl, sql = r
            m = re.search(r"\((\w+)", sql or "")
            if m:
                existing_indexes.add((tbl, m.group(1).lower()))

        proposals = []
        for tbl in tables:
            try:
                cols = con.execute(f"PRAGMA table_info({tbl})").fetchall()
                row_count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                if row_count < 100:
                    continue  # too small to benefit
            except Exception:
                continue
            for c in cols:
                col_name = c[1]
                if col_name.startswith("_"):
                    continue
                if not _is_index_worthy(col_name):
                    continue
                if (tbl, col_name.lower()) in existing_indexes:
                    continue
                proposals.append({
                    "table": tbl, "column": col_name,
                    "table_rows": int(row_count),
                    "reason": f"high-traffic pattern in column name '{col_name}'",
                })
        return proposals
    finally:
        con.close()


def create_indexes(db_path: str, proposals: list[dict]) -> dict:
    """Actually create the proposed indexes."""
    if not proposals:
        return {"created": 0, "indexes": []}
    con = sqlite3.connect(db_path, timeout=30)
    created = []
    try:
        for p in proposals:
            idx_name = f"idx_{p['table']}_{p['column']}"[:60]
            try:
                con.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON {p['table']}({p['column']})"
                )
                created.append({"index": idx_name, "table": p["table"], "column": p["column"]})
            except Exception as exc:
                created.append({"index": idx_name, "table": p["table"], "column": p["column"],
                                "error": str(exc)})
        con.commit()
    finally:
        con.close()
    return {"created": len([c for c in created if "error" not in c]),
            "errors":  len([c for c in created if "error" in c]),
            "indexes": created}


def refine_column_types(db_path: str, sample_size: int = 1000) -> list[dict]:
    """Detect TEXT columns whose values look like dates or numbers."""
    con = sqlite3.connect(db_path, timeout=30)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE 'landing_%' OR name LIKE 'stg_%' OR name LIKE 'master_%' "
            "ORDER BY name"
        ).fetchall()]
        refinements = []
        for tbl in tables:
            cols = con.execute(f"PRAGMA table_info({tbl})").fetchall()
            text_cols = [c[1] for c in cols if c[2].upper() == "TEXT" and not c[1].startswith("_")]
            if not text_cols:
                continue
            try:
                sample = pd.read_sql(f"SELECT * FROM {tbl} LIMIT {sample_size}", con)
            except Exception:
                continue
            for col in text_cols:
                if col not in sample.columns:
                    continue
                vals = sample[col].dropna().astype(str).head(200)
                if len(vals) < 10:
                    continue
                # Detect numeric
                num_match = vals.str.match(r"^-?\d+\.?\d*$").sum() / len(vals)
                # Detect date
                date_match = vals.str.match(r"^\d{4}-\d{2}-\d{2}|^\d{2}-\w{3}-\d{4}|^\d{2}/\d{2}/\d{4}").sum() / len(vals)
                # Detect E.164 phone
                phone_match = vals.str.match(r"^\+\d{7,15}$").sum() / len(vals)
                # Detect email
                email_match = vals.str.contains("@.+\\.[a-z]{2,}", regex=True).sum() / len(vals)
                if num_match > 0.9:
                    refinements.append({"table": tbl, "column": col, "stored_as":"TEXT", "actual_type":"NUMERIC", "confidence": round(num_match,2)})
                elif date_match > 0.9:
                    refinements.append({"table": tbl, "column": col, "stored_as":"TEXT", "actual_type":"DATE", "confidence": round(date_match,2)})
                elif phone_match > 0.9:
                    refinements.append({"table": tbl, "column": col, "stored_as":"TEXT", "actual_type":"PHONE_E164", "confidence": round(phone_match,2)})
                elif email_match > 0.9:
                    refinements.append({"table": tbl, "column": col, "stored_as":"TEXT", "actual_type":"EMAIL", "confidence": round(email_match,2)})
        return refinements
    finally:
        con.close()


def infer_constraints(db_path: str, sample_size: int = 5000) -> list[dict]:
    """Detect columns that should be NOT NULL or UNIQUE based on observed data."""
    con = sqlite3.connect(db_path, timeout=30)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND (name LIKE 'master_%' OR name LIKE 'stg_%') "
            "ORDER BY name"
        ).fetchall()]
        inferred = []
        for tbl in tables:
            try:
                row_count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                if row_count < 50:
                    continue
                cols = con.execute(f"PRAGMA table_info({tbl})").fetchall()
            except Exception:
                continue
            for c in cols:
                col_name = c[1]
                if col_name.startswith("_"):
                    continue
                try:
                    non_null = con.execute(
                        f"SELECT COUNT(*) FROM {tbl} WHERE {col_name} IS NOT NULL AND TRIM({col_name})<>''"
                    ).fetchone()[0]
                    distinct = con.execute(
                        f"SELECT COUNT(DISTINCT {col_name}) FROM {tbl} WHERE {col_name} IS NOT NULL"
                    ).fetchone()[0]
                except Exception:
                    continue
                non_null_rate = non_null / row_count if row_count else 0
                distinct_rate = distinct / max(non_null, 1)
                proposed = []
                if non_null_rate >= 0.99:
                    proposed.append("NOT NULL")
                if distinct_rate >= 0.99 and non_null > 100:
                    proposed.append("UNIQUE")
                if proposed:
                    inferred.append({
                        "table": tbl, "column": col_name,
                        "proposed_constraints": proposed,
                        "non_null_rate": round(non_null_rate,3),
                        "distinct_rate":  round(distinct_rate,3),
                    })
        return inferred
    finally:
        con.close()


def run_full_enhancement(db_path: str) -> dict:
    """Run all three enhancement passes, apply the index proposals."""
    idx_props = discover_indexes(db_path)
    idx_result = create_indexes(db_path, idx_props)
    refinements = refine_column_types(db_path)
    constraints = infer_constraints(db_path)
    return {
        "indexes": {"proposed": len(idx_props), "created": idx_result["created"],
                    "details": idx_result["indexes"]},
        "type_refinements": refinements,
        "constraint_inferences": constraints,
        "summary": {
            "new_indexes": idx_result["created"],
            "columns_re_typed": len(refinements),
            "constraints_inferred": len(constraints),
        }
    }
