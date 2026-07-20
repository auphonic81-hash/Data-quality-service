"""Entity classifier — decides what kind of entity a dataframe represents.

Runs BEFORE the LLM column mapper. The LLM mapper is good at translating
columns within a known entity, but bad at deciding entity type from
overlapping signals. This module makes that decision deterministically
using PK pattern + column signature + status vocabulary.

Why this matters: without it, sales orders (which have customer_id, amount,
status, date) get classified as invoices, polluting master_invoices
with 5,000+ rows that don't belong there.

Design:
- Each candidate entity has a "fingerprint" — patterns and column hints
- Score each fingerprint against the incoming dataframe
- Return the highest-scoring entity above the confidence threshold
- Below threshold, return None and let LLM/AutoNormalize handle it

Pure stdlib (re) + pandas. No custom rule engine, no ML model.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Fingerprints — declarative rules for each entity type
# ---------------------------------------------------------------------------

@dataclass
class EntityFingerprint:
    """Declarative signature for one entity type."""
    entity: str
    pk_patterns: list[str] = field(default_factory=list)        # regex list, OR-matched
    required_columns: list[str] = field(default_factory=list)   # column names (lowercase, partial-match)
    distinctive_columns: list[str] = field(default_factory=list)  # high-signal column names
    status_values: list[str] = field(default_factory=list)      # status vocabulary
    forbidden_columns: list[str] = field(default_factory=list)  # if present, lowers score


FINGERPRINTS = [
    EntityFingerprint(
        entity="SalesOrders",
        pk_patterns=[r"^SO-\d+$", r"^ORD-\d+$", r"^O-\d+$"],
        distinctive_columns=["order_id", "qty", "quantity", "product_id", "total_amt"],
        status_values=["Open", "Shipped", "Delivered", "Cancelled", "Returned"],
    ),
    EntityFingerprint(
        entity="Invoices",
        pk_patterns=[r"^(INV|BILL)-(\d{4}-)?\d+$"],
        required_columns=["invoice"],  # invoice_no or invoice_id or invoice_date
        distinctive_columns=["invoice_no", "invoice_id", "due_date", "bill_type"],
        status_values=["Paid", "Unpaid", "Open", "Overdue", "Pending", "Cancelled", "Refunded", "Draft"],
        forbidden_columns=["qty", "quantity", "product_id"],  # invoices don't have these
    ),
    EntityFingerprint(
        entity="Shipments",
        pk_patterns=[r"^SHP-\d+$", r"^TRK-?\d+$"],
        distinctive_columns=["ship_id", "tracking_no", "shipped_dt", "carrier"],
        status_values=["Shipped", "Delivered", "In Transit", "Processing", "Returned"],
    ),
    EntityFingerprint(
        entity="Customers",
        pk_patterns=[r"^C-\d{4,6}$"],
        distinctive_columns=["customer_id", "full_name", "email"],
        status_values=["Active", "Inactive", "Pending", "Suspended"],
        # These columns indicate transactional / line-item data, not customer master.
        # An orders.csv that includes denormalized customer columns should NOT
        # be classified as Customers because of those columns.
        forbidden_columns=[
            "invoice_no", "invoice_id", "bill_type",
            "order", "order_id", "order_number",
            "product_id", "product_sku", "product_name", "product_category",
            "quantity", "qty", "unit_price", "total", "total_amt",
            "tax", "discount", "payment_method", "shipped_dt", "ship_id",
        ],
    ),
    EntityFingerprint(
        entity="Vendors",
        pk_patterns=[r"^V-\d{4,6}$"],
        distinctive_columns=["vendor_id", "vendor_name", "contact_phone"],
        status_values=["Active", "Inactive"],
        forbidden_columns=[
            "full_name", "invoice_no", "order", "order_id", "order_number",
            "product_sku", "quantity", "qty",
        ],
    ),
    EntityFingerprint(
        entity="Employees",
        pk_patterns=[r"^E-\d{4,6}$", r"^EMP-\d+$"],
        distinctive_columns=["employee_id", "emp_id", "department_code", "salary"],
        status_values=["Active", "Inactive", "Terminated", "On Leave"],
    ),
    EntityFingerprint(
        entity="Products",
        pk_patterns=[r"^SKU-[A-Z0-9-]+$", r"^P-\d+$"],
        distinctive_columns=["sku", "product_name", "msrp", "category"],
    ),
    EntityFingerprint(
        entity="Payments",
        pk_patterns=[r"^PMT-\d+$"],
        distinctive_columns=["pmt_id", "pmt_amt", "reference_no", "payment_method"],
    ),
    EntityFingerprint(
        entity="Opportunities",
        pk_patterns=[r"^OPP-\d+$"],
        distinctive_columns=["opp_id", "opp_name", "close_dt", "owner_emp_id"],
    ),
    EntityFingerprint(
        entity="Contracts",
        pk_patterns=[r"^CT-\d+$"],
        distinctive_columns=["contract_id", "contract_value", "end_dt"],
    ),
    EntityFingerprint(
        entity="Accounts",
        pk_patterns=[r"^A-\d{4,6}$"],
        distinctive_columns=["acct_id", "account_name", "annual_revenue", "employee_count"],
    ),
    EntityFingerprint(
        entity="Locations",
        pk_patterns=[r"^LOC-\d+$"],
        distinctive_columns=["loc_id", "city", "country"],
    ),
    EntityFingerprint(
        entity="Departments",
        pk_patterns=[r"^D-\d+$"],
        distinctive_columns=["department_code", "dept_name", "cost_center"],
    ),
    EntityFingerprint(
        entity="Inventory",
        pk_patterns=[r"^INVT-\d+$"],
        distinctive_columns=["inv_item_id", "last_count_dt", "quantity"],
    ),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 50  # below this, return None — let LLM/AutoNormalize decide


def _find_pk_column(df: pd.DataFrame) -> Optional[str]:
    """Best guess at the PK column — highest uniqueness ratio.

    A PK should have ~1.0 uniqueness (one value per row). A foreign key
    like customer_id in an orders table has lower uniqueness (~0.5).
    So uniqueness is the primary signal; ID-like naming is a tiebreaker.
    """
    return _find_pk_candidates(df, top_n=1)[0] if not df.empty else None


def _find_pk_candidates(df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """Return top-N PK candidates ranked by (uniqueness, id_like_naming).

    Considers ALL columns with uniqueness >= 0.9. ID-naming bumps a tied
    candidate up. This lets the scorer try multiple PKs per fingerprint.
    """
    if df.empty:
        return []
    candidates = []
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        uniqueness = non_null.nunique() / len(non_null)
        if uniqueness < 0.9:
            continue
        lower = col.lower()
        id_like = (lower.endswith("_id") or lower.endswith("_no") or
                   lower.endswith("_number") or lower.endswith("_num") or
                   lower.endswith("_code") or lower == "id" or lower == "sku" or
                   lower.endswith("_sku"))
        candidates.append((col, uniqueness, id_like))
    if not candidates:
        # Fallback: first column as a last resort
        return [df.columns[0]] if len(df.columns) else []
    # Rank: ID-like first, then uniqueness desc
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return [c[0] for c in candidates[:top_n]]


def _score_fingerprint(fp: EntityFingerprint, df: pd.DataFrame, pk_candidates: list[str]) -> int:
    """Compute confidence score for one fingerprint against the dataframe.

    Tries EACH PK candidate against the fingerprint's PK patterns and uses
    the best match. This fixes the wide-table problem where the "wrong"
    PK (a foreign key like customer_id) was tested against SalesOrders
    patterns and naturally failed.

    Weighting (max 100):
      - PK pattern match: 50 points (any candidate matches)
      - Distinctive column match: 10 points each (cap 30)
      - Status vocabulary match: 15 points
      - Required column match: 5 points each
      - Forbidden column penalty: -20 each (substring or prefix match)
    """
    score = 0
    cols_lower = [c.lower() for c in df.columns]

    # PK pattern check — try every PK candidate
    if pk_candidates and fp.pk_patterns:
        best_match = False
        for pk_col in pk_candidates:
            if pk_col not in df.columns:
                continue
            sample = df[pk_col].dropna().astype(str).head(20).tolist()
            if not sample:
                continue
            for pattern in fp.pk_patterns:
                matches = sum(1 for v in sample if re.match(pattern, v))
                if matches / len(sample) > 0.8:
                    best_match = True
                    break
            if best_match:
                break
        if best_match:
            score += 50

    # Distinctive columns
    distinctive_hits = 0
    for col_hint in fp.distinctive_columns:
        if any(col_hint in c for c in cols_lower):
            distinctive_hits += 1
    score += min(30, distinctive_hits * 10)

    # Status vocabulary
    status_col = next((c for c in df.columns if "status" in c.lower()), None)
    if status_col and fp.status_values:
        observed = set(df[status_col].dropna().astype(str).str.strip().unique())
        expected = set(fp.status_values)
        if observed and observed.issubset(expected):
            score += 15
        elif observed and observed & expected:
            score += 5  # partial match

    # Required columns
    for col_hint in fp.required_columns:
        if any(col_hint in c for c in cols_lower):
            score += 5

    # Forbidden column penalty — also catch prefix variations
    # ("order" hint matches order_number, order_id, order_date, etc.)
    for col_hint in fp.forbidden_columns:
        hit = False
        for c in cols_lower:
            if col_hint in c or c.startswith(col_hint + "_") or c == col_hint:
                hit = True
                break
        if hit:
            score -= 20

    return score


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def classify_entity(df: pd.DataFrame, filename: Optional[str] = None) -> tuple[Optional[str], int, dict]:
    """Classify what entity this dataframe represents.

    Returns (entity_name, confidence_score, debug_info).
    Returns (None, top_score, debug_info) if no entity scored above threshold.

    Args:
        df: the dataframe to classify
        filename: optional source filename, used as a tiebreaker hint

    Example:
        entity, conf, info = classify_entity(df)
        if entity:
            print(f"Classified as {entity} with confidence {conf}")
        else:
            print(f"Low confidence, top guess was {info['top']}")
    """
    if df.empty or len(df.columns) == 0:
        return None, 0, {"reason": "empty dataframe"}

    # Get top PK candidates instead of just one — wide tables have multiple
    # high-uniqueness columns (order_number AND customer_id), and we want to
    # try each against every fingerprint.
    pk_candidates = _find_pk_candidates(df, top_n=5)
    pk_col = pk_candidates[0] if pk_candidates else None

    scores = []
    for fp in FINGERPRINTS:
        s = _score_fingerprint(fp, df, pk_candidates)
        scores.append((fp.entity, s))

    # Filename hint as tiebreaker (small bonus, only if filename strongly suggests an entity)
    if filename:
        fn_lower = filename.lower()
        filename_hints = {
            "order": "SalesOrders", "sales": "SalesOrders",
            "invoice": "Invoices", "bill": "Invoices",
            "shipment": "Shipments", "ship": "Shipments",
            "customer": "Customers", "vendor": "Vendors", "supplier": "Vendors",
            "employee": "Employees", "product": "Products",
            "payment": "Payments", "opportun": "Opportunities",
            "contract": "Contracts", "account": "Accounts",
            "location": "Locations", "department": "Departments",
            "inventory": "Inventory",
        }
        for hint, entity in filename_hints.items():
            if hint in fn_lower:
                # +10 boost only if score is already non-trivial
                scores = [(e, s + 10 if e == entity and s > 20 else s) for e, s in scores]
                break

    scores.sort(key=lambda x: x[1], reverse=True)
    top_entity, top_score = scores[0]

    debug = {
        "pk_column": pk_col,
        "top_3": scores[:3],
        "threshold": CONFIDENCE_THRESHOLD,
    }

    if top_score < CONFIDENCE_THRESHOLD:
        debug["reason"] = f"top score {top_score} below threshold {CONFIDENCE_THRESHOLD}"
        return None, top_score, debug

    return top_entity, top_score, debug