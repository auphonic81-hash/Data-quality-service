"""Splink-based deduplication — Fellegi-Sunter probabilistic record linkage.

Replaces custom string-similarity dedup with the same math Informatica MDM Hub
uses underneath. Used by gov.uk, US Census, and major enterprises.

Two modes:
1. dedupe(df, entity_type) - find duplicates within one DataFrame
2. link(df_a, df_b, entity_type) - find matches between two sources

Returns clusters of matched rows + match probability per pair.
"""
from __future__ import annotations
import pandas as pd
from typing import Any

# Lazy imports so this module doesn't crash on startup if splink has issues
def _get_splink():
    import splink
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    import splink.comparison_library as cl
    return splink, DuckDBAPI, Linker, SettingsCreator, block_on, cl


# Per-entity comparison strategies (what columns to compare, how)
ENTITY_BLUEPRINTS: dict[str, dict[str, Any]] = {
    "Customers": {
        "unique_id_column": "_splink_id",
        "comparisons_for": ["full_name", "email", "phone", "country"],
        "blocking_columns": ["country"],  # only compare records within same country
    },
    "Employees": {
        "unique_id_column": "_splink_id",
        "comparisons_for": ["full_name", "email", "phone", "department_code"],
        "blocking_columns": ["country"],
    },
    "Vendors": {
        "unique_id_column": "_splink_id",
        "comparisons_for": ["vendor_name", "contact_email", "contact_phone"],
        "blocking_columns": ["country"],
    },
    "Invoices": {
        "unique_id_column": "_splink_id",
        "comparisons_for": ["invoice_no", "customer_id", "amount"],
        "blocking_columns": ["customer_id"],
    },
    "Products": {
        "unique_id_column": "_splink_id",
        "comparisons_for": ["product_name", "sku", "category"],
        "blocking_columns": ["category"],
    },
}


def _build_settings(blueprint: dict, df: pd.DataFrame):
    """Build a Splink settings config from a blueprint."""
    splink, DuckDBAPI, Linker, SettingsCreator, block_on, cl = _get_splink()

    comparisons = []
    for col in blueprint["comparisons_for"]:
        if col not in df.columns:
            continue
        # Use ExactMatch + LevenshteinAtThresholds for name/email/phone fields
        if col in ("full_name", "vendor_name", "product_name"):
            comparisons.append(cl.NameComparison(col))
        elif col in ("email", "contact_email"):
            comparisons.append(cl.EmailComparison(col))
        elif col in ("phone", "contact_phone", "phone_no"):
            comparisons.append(cl.ExactMatch(col))
        else:
            comparisons.append(cl.ExactMatch(col))

    blocking_rules = []
    for col in blueprint.get("blocking_columns", []):
        if col in df.columns:
            blocking_rules.append(block_on(col))
    if not blocking_rules:
        # No blocking column means we'd compare every pair — too expensive.
        # Pick a blocking column heuristically (first non-id column).
        for col in df.columns:
            if col not in ("_splink_id",) and not col.startswith("_"):
                blocking_rules.append(block_on(col))
                break

    return SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name=blueprint["unique_id_column"],
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
        retain_intermediate_calculation_columns=False,
    )


def dedupe(df: pd.DataFrame, entity_type: str, threshold: float = 0.95) -> pd.DataFrame:
    """Find duplicates within ONE DataFrame using probabilistic linkage.

    Returns a DataFrame with cluster IDs — rows sharing a cluster_id are duplicates.
    """
    if entity_type not in ENTITY_BLUEPRINTS:
        return pd.DataFrame()

    if df.empty or len(df) < 2:
        return pd.DataFrame()

    blueprint = ENTITY_BLUEPRINTS[entity_type]
    splink, DuckDBAPI, Linker, SettingsCreator, block_on, cl = _get_splink()

    # Splink needs a unique id column
    work_df = df.copy().reset_index(drop=True)
    work_df["_splink_id"] = work_df.index.astype(str)

    settings = _build_settings(blueprint, work_df)
    linker = Linker(work_df, settings, db_api=DuckDBAPI())

    # Use Splink's unsupervised parameter estimation
    try:
        linker.training.estimate_probability_two_random_records_match(
            deterministic_matching_rules=[
                block_on(blueprint.get("blocking_columns", [df.columns[0]])[0])
            ],
            recall=0.7,
        )
        linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    except Exception:
        # Fall back to default parameters if estimation fails
        pass

    # Predict match probabilities
    predictions = linker.inference.predict()
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_probability=threshold
    )

    result = clusters.as_pandas_dataframe()
    return result


def link(df_a: pd.DataFrame, df_b: pd.DataFrame, entity_type: str,
         threshold: float = 0.9) -> pd.DataFrame:
    """Find matches BETWEEN two DataFrames (different sources for the same entity).

    Returns pairs of matched rows with probability scores.
    """
    if entity_type not in ENTITY_BLUEPRINTS:
        return pd.DataFrame()
    if df_a.empty or df_b.empty:
        return pd.DataFrame()

    blueprint = ENTITY_BLUEPRINTS[entity_type]
    splink, DuckDBAPI, Linker, SettingsCreator, block_on, cl = _get_splink()

    a = df_a.copy().reset_index(drop=True)
    a["_splink_id"] = "A_" + a.index.astype(str)
    a["source"] = "A"

    b = df_b.copy().reset_index(drop=True)
    b["_splink_id"] = "B_" + b.index.astype(str)
    b["source"] = "B"

    settings = _build_settings(blueprint, a)
    # Override link_type for cross-source matching
    settings.link_type = "link_only"

    linker = Linker([a, b], settings, db_api=DuckDBAPI())

    try:
        linker.training.estimate_probability_two_random_records_match(
            deterministic_matching_rules=[
                block_on(blueprint.get("blocking_columns", [df_a.columns[0]])[0])
            ],
            recall=0.7,
        )
        linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    except Exception:
        pass

    predictions = linker.inference.predict()
    df_pairs = predictions.as_pandas_dataframe()
    df_pairs = df_pairs[df_pairs["match_probability"] >= threshold]
    return df_pairs


def info() -> str:
    splink, *_ = _get_splink()
    return f"Splink v{splink.__version__} (Fellegi-Sunter probabilistic linkage)"


def find_fuzzy_duplicates_in_master(db_path: str, entity_type: str,
                                     threshold: float = 0.95) -> dict:
    """Scan a master_* table for fuzzy duplicates that exact match missed.

    Returns clusters of records that Splink thinks are the same entity
    despite not being caught by exact PK match. These are flagged for review,
    NOT auto-merged — same as Informatica's match-review queue.
    """
    import sqlite3, pandas as pd

    table_map = {
        "Customers":"master_customers", "Vendors":"master_vendors",
        "Products":"master_products",   "Invoices":"master_invoices",
        "Employees":"master_employees",
    }
    table = table_map.get(entity_type)
    if not table:
        return {"error": f"Unknown entity {entity_type}"}

    con = sqlite3.connect(db_path, timeout=30)
    try:
        df = pd.read_sql(f"SELECT * FROM {table} WHERE archived_at IS NULL", con)
    finally:
        con.close()

    if len(df) < 2:
        return {"entity": entity_type, "rows_scanned": len(df), "clusters": []}

    try:
        clusters_df = dedupe(df, entity_type, threshold=threshold)
    except Exception as exc:
        return {"entity": entity_type, "error": str(exc)}

    if clusters_df.empty:
        return {"entity": entity_type, "rows_scanned": len(df),
                "fuzzy_duplicates_found": 0, "clusters": []}

    # Find clusters with 2+ members (the actual duplicates)
    if "cluster_id" not in clusters_df.columns:
        return {"entity": entity_type, "rows_scanned": len(df), "clusters": [],
                "note": "Splink returned no cluster_id column"}

    cluster_sizes = clusters_df["cluster_id"].value_counts()
    duplicate_clusters = cluster_sizes[cluster_sizes >= 2].index.tolist()

    samples = []
    for cid in duplicate_clusters[:20]:  # cap output
        members = clusters_df[clusters_df["cluster_id"] == cid]
        samples.append({
            "cluster_id": str(cid),
            "size": int(len(members)),
            "member_indices": members["_splink_id"].tolist()[:5],
        })

    return {
        "entity": entity_type,
        "rows_scanned": int(len(df)),
        "fuzzy_duplicates_found": int(sum(cluster_sizes[cluster_sizes >= 2])),
        "duplicate_clusters": int(len(duplicate_clusters)),
        "sample_clusters": samples,
        "threshold": threshold,
        "engine": "Splink (Fellegi-Sunter)",
    }


def link_cross_source(db_path: str, source_a_table: str, source_b_table: str,
                       entity_type: str, threshold: float = 0.9) -> dict:
    """Link records between two landing tables using probabilistic matching.

    Use case: same employee in REST API and SQLite ERP, but with slight differences
    (typo in name, different email format). Splink finds the matches that exact
    PK comparison misses.
    """
    import sqlite3, pandas as pd

    con = sqlite3.connect(db_path, timeout=30)
    try:
        df_a = pd.read_sql(f"SELECT * FROM {source_a_table}", con)
        df_b = pd.read_sql(f"SELECT * FROM {source_b_table}", con)
    finally:
        con.close()

    # Strip provenance cols
    df_a = df_a[[c for c in df_a.columns if not c.startswith("_landing_")]].copy()
    df_b = df_b[[c for c in df_b.columns if not c.startswith("_landing_")]].copy()

    # Normalize column names to match the blueprint (lowercase, canonical names)
    # Each table has different cryptic column names — try common canonicalization
    def canonicalize(df):
        rename = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ("emp_no","emp_id","employee_id"):       rename[c] = "employee_id"
            elif lc in ("full_nm","full_name","emp_full_name"): rename[c] = "full_name"
            elif lc in ("work_email","email_addr","email"):    rename[c] = "email"
            elif lc in ("mobile_nr","phone_nr","phone"):       rename[c] = "phone"
            elif lc in ("dept_cd","dept","department_code"):   rename[c] = "department_code"
            elif lc in ("ctry","country","location"):          rename[c] = "country"
            elif lc in ("customer_id","cust_id","cust_key"):   rename[c] = "customer_id"
            elif lc in ("contact_nm","customer_name"):         rename[c] = "full_name"
        return df.rename(columns=rename)

    df_a = canonicalize(df_a)
    df_b = canonicalize(df_b)

    blueprint = ENTITY_BLUEPRINTS.get(entity_type)
    if not blueprint:
        return {"error": f"Unknown entity {entity_type}"}

    splink, DuckDBAPI, Linker, SettingsCreator, block_on, cl = _get_splink()

    # Find common columns + automatically add ID columns as comparisons
    common_cols = [c for c in blueprint["comparisons_for"] if c in df_a.columns and c in df_b.columns]
    for id_col in ("employee_id","customer_id","country"):
        if id_col in df_a.columns and id_col in df_b.columns and id_col not in common_cols:
            common_cols.append(id_col)
    if len(common_cols) < 2:
        return {"error": f"Not enough common columns. Found: {common_cols}",
                "df_a_cols": list(df_a.columns), "df_b_cols": list(df_b.columns)}

    # Restrict both dataframes to columns common to BOTH — Splink unions them
    shared = [c for c in df_a.columns if c in df_b.columns]
    df_a = df_a[shared].copy()
    df_b = df_b[shared].copy()
    # Add unique IDs prefixed by source
    df_a = df_a.reset_index(drop=True); df_a["_splink_id"] = "A_" + df_a.index.astype(str); df_a["source_dataset"] = source_a_table
    df_b = df_b.reset_index(drop=True); df_b["_splink_id"] = "B_" + df_b.index.astype(str); df_b["source_dataset"] = source_b_table

    # Build comparisons only for common columns
    comparisons = []
    for col in common_cols:
        if col in ("full_name",):
            comparisons.append(cl.NameComparison(col))
        elif col in ("email",):
            comparisons.append(cl.EmailComparison(col))
        else:
            comparisons.append(cl.ExactMatch(col))

    # Block on a COARSER group (department, country) so Splink has real cross-pairs.
    # If we block on employee_id, only identical IDs get compared — but those are
    # already exact matches and Splink has nothing to add. We want Splink to find
    # PROBABILISTIC matches across the broader group.
    block_col = None
    for candidate in ("department_code","country","status"):
        if candidate in df_a.columns and candidate in df_b.columns:
            block_col = candidate
            break
    # Fall back to first shared column
    if not block_col:
        for c in df_a.columns:
            if c in df_b.columns and not c.startswith("_"):
                block_col = c
                break
    if not block_col:
        return {"error": "No suitable blocking column"}

    settings = SettingsCreator(
        link_type="link_only",
        unique_id_column_name="_splink_id",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=[block_on(block_col)],
    )

    linker = Linker([df_a, df_b], settings, db_api=DuckDBAPI())

    try:
        linker.training.estimate_probability_two_random_records_match(
            deterministic_matching_rules=[block_on(block_col)], recall=0.9,
        )
        linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    except Exception: pass

    predictions = linker.inference.predict()
    df_pairs = predictions.as_pandas_dataframe()

    matches = df_pairs[df_pairs["match_probability"] >= threshold]
    samples = []
    for _, row in matches.head(10).iterrows():
        samples.append({
            "match_probability": float(row["match_probability"]),
            "source_a_id": str(row["_splink_id_l"]),
            "source_b_id": str(row["_splink_id_r"]),
            **{c: str(row.get(c+"_l","")) for c in common_cols},
        })

    return {
        "engine": "Splink (Fellegi-Sunter)",
        "source_a": source_a_table, "source_a_rows": int(len(df_a)),
        "source_b": source_b_table, "source_b_rows": int(len(df_b)),
        "common_comparison_columns": common_cols,
        "blocking_column": block_col,
        "matches_at_threshold": int(len(matches)),
        "threshold": threshold,
        "sample_matches": samples,
    }
