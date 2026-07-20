"""Auto-onboard new entity types when the deterministic classifier can't match.

When entity_classifier returns confidence < CONFIDENCE_THRESHOLD, this module
asks the LLM to propose a canonical entity: name, primary key column, field map.
Guardrails reject nonsense proposals before any master table is created.

Persistent state lives in `dynamic_entities` table so newly-onboarded entities
survive Flask restarts.
"""
from __future__ import annotations
import json
import re
import sqlite3
from typing import Any

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
LLM_TIMEOUT = 90

MIN_ROWS = 10
MIN_COLUMNS = 3
MAX_AUTO_ENTITIES_PER_SESSION = 5

FORBIDDEN_NAMES = {
    "temp", "test", "tmp", "unknown", "unnamed", "data",
    "table", "misc", "other", "random", "garbage",
}

NAME_PATTERN = re.compile(r"^[A-Z][a-zA-Z]{2,30}$")


def ensure_dynamic_entities_table(db_path: str) -> None:
    with sqlite3.connect(db_path, timeout=30) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS dynamic_entities (
            entity_name         TEXT PRIMARY KEY,
            master_table        TEXT NOT NULL,
            primary_key         TEXT NOT NULL,
            field_map_json      TEXT NOT NULL,
            column_signature    TEXT NOT NULL,
            confidence          REAL,
            created_from_file   TEXT,
            created_at          TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """)


class EntityProposer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        ensure_dynamic_entities_table(db_path)

    def propose(self, df: pd.DataFrame, filename: str | None = None,
                existing_entities: dict | None = None) -> dict:
        if len(df) < MIN_ROWS:
            return {"status": "rejected", "reason": f"only {len(df)} rows, need >={MIN_ROWS}"}
        if len(df.columns) < MIN_COLUMNS:
            return {"status": "rejected", "reason": f"only {len(df.columns)} columns, need >={MIN_COLUMNS}"}

        session_count = self._count_recent_auto_entities()
        if session_count >= MAX_AUTO_ENTITIES_PER_SESSION:
            return {"status": "rejected", "reason": f"session limit reached ({MAX_AUTO_ENTITIES_PER_SESSION})"}

        signature = self._column_signature(df)
        existing_match = self._find_by_signature(signature)
        if existing_match:
            # This entity was already onboarded — route to it instead of creating duplicate
            return {
                "status": "route_to_existing",
                "reason": f"columns match previously-onboarded entity '{existing_match}'",
                "route_to_entity": existing_match,
            }

        try:
            raw = self._call_llm(df, filename, existing_entities or {})
        except Exception as e:
            return {"status": "rejected", "reason": f"LLM error: {e}"}

        parsed = self._parse_llm_response(raw)
        if "error" in parsed:
            return {"status": "rejected", "reason": parsed["error"]}

        # Provenance/audit columns leaked from landing tables should NOT be in canonical schema
        _skip_prefixes = ("_landing_", "landing_", "_source_", "source_dataset_", "source_system", "_created_", "_promoted_", "ingested_at", "last_updated", "merged_from_count", "confidence_score", "archived_at")
        _skip_exact = {"landing_source_system", "landing_ingested_at", "source_system", "source_dataset_id",
                       "ingested_at", "last_updated", "merged_from_count", "confidence_score",
                       "archived_at", "_source_system", "_promoted_from", "_created_at", "source_row_index"}
        # Purge audit/landing columns the LLM may have proposed as canonical
        parsed["field_map"] = {
            k: v for k, v in parsed["field_map"].items()
            if k.lower() not in _skip_exact and not any(k.lower().startswith(pfx) for pfx in _skip_prefixes)
        }

        # Backfill: any source column not mapped by LLM gets identity mapping
        # (canonical_name = source_name). LLMs often drop columns silently.
        mapped_sources = set()
        for _canon, _aliases in parsed["field_map"].items():
            if isinstance(_aliases, list):
                mapped_sources.update(str(a).lower() for a in _aliases)
            else:
                mapped_sources.add(str(_aliases).lower())
        for _src_col in df.columns:
            _src_lower = str(_src_col).lower()
            if _src_lower in _skip_exact or any(_src_lower.startswith(pfx) for pfx in _skip_prefixes):
                continue
            if _src_lower in mapped_sources:
                continue
            parsed["field_map"][_src_lower] = [_src_lower]
            mapped_sources.add(_src_lower)

        name = parsed["entity_name"]
        if name.lower() in FORBIDDEN_NAMES:
            return {"status": "rejected", "reason": f"forbidden name '{name}'"}
        if not NAME_PATTERN.match(name):
            return {"status": "rejected", "reason": f"invalid name format '{name}'"}
        if existing_entities and name in existing_entities:
            return {"status": "rejected", "reason": f"name '{name}' already exists"}

        pk = parsed["primary_key"]
        if pk not in df.columns:
            lower_map = {c.lower(): c for c in df.columns}
            if pk.lower() in lower_map:
                pk = lower_map[pk.lower()]
                parsed["primary_key"] = pk
            else:
                return {"status": "rejected", "reason": f"proposed PK '{pk}' not in columns"}

        pk_uniqueness = df[pk].nunique() / max(1, len(df))
        if pk_uniqueness < 0.9:
            return {"status": "rejected", "reason": f"PK '{pk}' only {pk_uniqueness:.0%} unique, need >=90%"}

        return {
            "status": "proposed",
            "proposal": {
                "entity_name": name,
                "primary_key": pk,
                "field_map": parsed["field_map"],
                "confidence": float(parsed.get("confidence", 0.8)),
                "column_signature": signature,
            },
        }

    def _call_llm(self, df: pd.DataFrame, filename: str | None,
                  existing_entities: dict) -> str:
        col_summary = []
        for col in df.columns[:15]:
            samples = df[col].dropna().astype(str).head(3).tolist()
            col_summary.append(f"  - {col} (samples: {samples})")

        existing_names = list(existing_entities.keys()) if existing_entities else []
        # Derive filename hint - strip extension + common prefixes
        _fname_stem = ""
        if filename:
            _fname_stem = filename.rsplit(".", 1)[0].lower()
            for _pfx in ("api_", "sqlite_", "file_", "landing_", "stg_", "master_"):
                if _fname_stem.startswith(_pfx):
                    _fname_stem = _fname_stem[len(_pfx):]

        prompt = f"""You are onboarding a new entity type into a Master Data Management system.

FILE: {filename or "unknown"}
FILENAME HINT (strongest signal for entity name): {_fname_stem or "n/a"}

COLUMNS ({len(df.columns)} total, first 15 shown):
{chr(10).join(col_summary)}

ROW COUNT: {len(df)}

EXISTING ENTITIES (do NOT propose any of these):
{", ".join(existing_names) if existing_names else "(none)"}

CRITICAL RULES:
1. entity_name MUST be derived from the FILE NAME, not from any single column.
   Example: file "grievances.csv" -> entity_name "Grievances"
   Example: file "warranties.csv" -> entity_name "Warranties"
   Example: file "insurance_policies.csv" -> entity_name "InsurancePolicies"
2. Do NOT pick a column name as the entity (e.g., "department_filed" is an attribute, not the entity).
3. Do NOT propose an entity name that matches EXISTING ENTITIES list. Choose a different noun.
4. entity_name format: PascalCase business noun. NEVER "temp", "test", "data", "unknown", "misc", "table".
5. primary_key: pick the column whose values look like unique identifiers (IDs, codes, numbers).
6. field_map: map every business column to a canonical name -> aliases list.
7. Return ONLY valid JSON. No prose. No markdown.

Response shape (exact JSON):
{{"entity_name":"Grievances","primary_key":"grievance_id","field_map":{{"grievance_id":["grievance_id"],"complainant_name":["complainant_name","name"],"department_filed":["department_filed","department"]}},"confidence":0.9}}"""

        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False, "format": "json"},
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _parse_llm_response(self, raw: str) -> dict:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return {"error": f"LLM returned non-JSON: {e}"}
        for k in ("entity_name", "primary_key", "field_map"):
            if k not in data:
                return {"error": f"LLM response missing '{k}'"}
        if not isinstance(data["field_map"], dict) or not data["field_map"]:
            return {"error": "field_map must be non-empty dict"}
        return data

    @staticmethod
    def _column_signature(df: pd.DataFrame) -> str:
        return "|".join(sorted(str(c).lower() for c in df.columns))

    def _count_recent_auto_entities(self, hours: int = 24) -> int:
        with sqlite3.connect(self.db_path, timeout=30) as con:
            return con.execute(
                "SELECT COUNT(*) FROM dynamic_entities WHERE datetime(created_at) > datetime('now', ?)",
                (f"-{hours} hours",),
            ).fetchone()[0]

    def _find_by_signature(self, signature: str) -> str | None:
        with sqlite3.connect(self.db_path, timeout=30) as con:
            row = con.execute(
                "SELECT entity_name FROM dynamic_entities WHERE column_signature = ?",
                (signature,),
            ).fetchone()
            return row[0] if row else None

    def register(self, proposal: dict, created_from_file: str | None = None) -> str:
        name = proposal["entity_name"]
        master_table = f"master_{name.lower()}"
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.execute(
                "INSERT OR REPLACE INTO dynamic_entities "
                "(entity_name, master_table, primary_key, field_map_json, "
                " column_signature, confidence, created_from_file) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    name, master_table, proposal["primary_key"],
                    json.dumps(proposal["field_map"]),
                    proposal["column_signature"],
                    float(proposal.get("confidence", 0.8)),
                    created_from_file,
                ),
            )
        return master_table


    # -----------------------------------------------------------------------
    # Table creation
    # -----------------------------------------------------------------------
    STANDARD_AUDIT_COLS = [
        "source_dataset_id TEXT",
        "source_row_index INTEGER",
        "ingested_at TEXT",
        "last_updated TEXT",
        "merged_from_count INTEGER DEFAULT 1",
        "confidence_score REAL DEFAULT 1.0",
        "archived_at TEXT",
        "_source_system TEXT",
        "_promoted_from TEXT",
        "_created_at TEXT",
    ]

    def create_master_table(self, proposal: dict) -> str:
        """Create master_<entity> table with canonical columns + audit columns.
        Returns the created table name.
        """
        name = proposal["entity_name"]
        pk = proposal["primary_key"]
        master_table = f"master_{name.lower()}"
        canonical_cols = list(proposal["field_map"].keys())

        # Reserved audit column names - canonical cols with these names are dropped
        _reserved = {"source_dataset_id", "source_row_index", "ingested_at",
                     "last_updated", "merged_from_count", "confidence_score",
                     "archived_at", "_source_system", "_promoted_from", "_created_at"}
        canonical_cols = [c for c in canonical_cols if c.lower() not in _reserved]
        # Also update the proposal so downstream code sees the filtered list
        proposal["field_map"] = {k: v for k, v in proposal["field_map"].items()
                                 if k.lower() not in _reserved}

        col_defs = []
        for col in canonical_cols:
            if col == pk:
                col_defs.append(f'"{col}" TEXT PRIMARY KEY')
            else:
                col_defs.append(f'"{col}" TEXT')
        col_defs.extend(self.STANDARD_AUDIT_COLS)

        create_sql = f'CREATE TABLE IF NOT EXISTS {master_table} ({", ".join(col_defs)})'
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.execute(create_sql)
        return master_table

    def create_and_register(self, proposal: dict, created_from_file: str | None = None) -> dict:
        """Create the master table AND persist in dynamic_entities.
        Returns a dict describing what happened so callers can log it.
        """
        master_table = self.create_master_table(proposal)
        self.register(proposal, created_from_file=created_from_file)
        return {
            "master_table": master_table,
            "entity_name": proposal["entity_name"],
            "primary_key": proposal["primary_key"],
            "columns_created": list(proposal["field_map"].keys()),
        }

    def load_all(self) -> list[dict]:
        """Load all persisted dynamic entities for startup registration."""
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM dynamic_entities").fetchall()
            out = []
            for r in rows:
                out.append({
                    "entity_name": r["entity_name"],
                    "master_table": r["master_table"],
                    "primary_key": r["primary_key"],
                    "field_map": json.loads(r["field_map_json"]),
                    "confidence": r["confidence"],
                    "created_at": r["created_at"],
                })
            return out
