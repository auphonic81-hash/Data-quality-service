from __future__ import annotations
import json
import sqlite3
import requests
from typing import Any


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
TIMEOUT = 60  # seconds per request

# Canonical master schema fields per entity — what the LLM should map to
CANONICAL_FIELDS = {
    "Customers": ["customer_id", "full_name", "email", "phone", "city", "country", "address", "status", "created_at"],
    "Vendors":   ["vendor_id", "vendor_name", "contact_phone", "country", "status"],
    "Products":  ["sku", "product_name", "category", "price", "msrp", "supplier_id", "stock"],
    "Invoices":  ["invoice_no", "customer_id", "amount", "due_date", "invoice_date", "status", "description"],
    "Employees": ["employee_id", "full_name", "email", "phone", "department_code", "department_name", "country", "hire_date", "status", "manager_id", "salary"],
}


class LLMColumnMapper:
    """Asks the local Ollama LLM to map source columns → master schema fields."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_cache()

    def _init_cache(self):
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS column_mapping_cache (
                cache_key     TEXT PRIMARY KEY,
                entity        TEXT NOT NULL,
                source_column TEXT NOT NULL,
                sample_values TEXT,
                suggested_master_field TEXT,
                confidence    REAL DEFAULT 0.0,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                source        TEXT DEFAULT 'llm'
            );
            CREATE INDEX IF NOT EXISTS idx_mapcache_entity
                ON column_mapping_cache(entity, source_column);
            """)

    # ─── Public API ──────────────────────────────────────────────────────

    def suggest(
        self,
        entity: str,
        source_column: str,
        sample_values: list[Any],
    ) -> dict[str, Any] | None:
        """Return a suggested master field for `source_column` or None.

        First checks cache. If not cached, calls Ollama, stores result.
        """
        if entity not in CANONICAL_FIELDS:
            return None

        # Normalize cache key — same column + entity always returns same answer
        cache_key = f"{entity.lower()}::{source_column.lower().strip()}"

        # Check cache
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row
            cached = con.execute(
                "SELECT suggested_master_field, confidence, source FROM column_mapping_cache WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
            if cached:
                return {
                    "master_field": cached["suggested_master_field"],
                    "confidence":   cached["confidence"],
                    "source":       cached["source"] + "_cached",
                }

        # Not cached — call LLM
        result = self._call_llm(entity, source_column, sample_values)
        if not result:
            return None

        # Cache result
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.execute(
                "INSERT OR REPLACE INTO column_mapping_cache "
                "(cache_key, entity, source_column, sample_values, suggested_master_field, confidence, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_key,
                    entity,
                    source_column,
                    json.dumps([str(v)[:50] for v in sample_values[:5]]),
                    result["master_field"],
                    result["confidence"],
                    "llm",
                ),
            )
        return result

    def get_cache_stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path, timeout=30) as con:
            con.row_factory = sqlite3.Row
            total = con.execute("SELECT COUNT(*) FROM column_mapping_cache").fetchone()[0]
            recent = con.execute(
                "SELECT entity, source_column, suggested_master_field, confidence, source, created_at "
                "FROM column_mapping_cache ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            return {"total": total, "recent": [dict(r) for r in recent]}

    # ─── LLM call ────────────────────────────────────────────────────────

    @staticmethod
    def _call_llm(entity: str, source_column: str, samples: list[Any]) -> dict[str, Any] | None:
        canonical = CANONICAL_FIELDS[entity]
        sample_str = ", ".join(repr(str(s)[:30]) for s in samples[:5] if s is not None and str(s).strip())

        # Build a numbered canonical field list so the LLM can\'t miss it
        canonical_listing = "\n".join([f"  {i+1}. {f}" for i, f in enumerate(canonical)])
        prompt = (
            f"Task: Match a source column to ONE field from a fixed list of canonical {entity} fields.\n\n"
            f"VALID CANONICAL FIELDS (you MUST pick exactly one of these, or NONE):\n{canonical_listing}\n\n"
            f"Source column name: \"{source_column}\"\n"
            f"Sample values from this column: [{sample_str}]\n\n"
            f"Rules:\n"
            f"- If the column maps to one of the valid fields above, return that exact field name.\n"
            f"- If no valid field matches, return \"NONE\".\n"
            f"- Use sample values AND column name to decide.\n\n"
            f"Output ONLY this JSON, nothing else:\n"
            f'{{"master_field": "<exact_field_from_list_or_NONE>", "confidence": <0.0-1.0>, "reason": "<one short sentence>"}}'
        )

        try:
            r = requests.post(
                OLLAMA_URL,
                json={
                    "model":  MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1, "num_ctx": 2048},
                },
                timeout=TIMEOUT,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            # qwen3.5 puts the JSON in "thinking" field, others use "response"
            raw = (data.get("response") or "").strip() or (data.get("thinking") or "").strip()
            master = None
            conf = 0.0
            reason = ""
            try:
                parsed = json.loads(raw)
                # Shape 1: {"master_field": "...", "confidence": ...}
                if isinstance(parsed, dict) and "master_field" in parsed:
                    master = (parsed.get("master_field") or "").strip()
                    conf = float(parsed.get("confidence", 0.7) or 0.7)
                    reason = parsed.get("reason", "") or ""
                # Shape 2: {"<source_col>": "<canonical_field>"}  — single key mapping
                elif isinstance(parsed, dict) and len(parsed) == 1:
                    val = list(parsed.values())[0]
                    if isinstance(val, str) and val.strip() and val.strip().upper() != "NONE":
                        master = val.strip()
                        conf = 0.85  # implicit confidence for direct mapping
                        reason = "inferred from {column → field} mapping"
                # Shape 3: {"mapping": "...", "field": "..."} or other nested
                elif isinstance(parsed, dict):
                    for k in ("field","mapped_field","canonical_field","target","result"):
                        if k in parsed and isinstance(parsed[k], str):
                            master = parsed[k].strip()
                            conf = float(parsed.get("confidence", 0.8) or 0.8)
                            break
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            # Fallback: scan the response for any canonical field name (chatty models)
            if not master or master.upper() == "NONE":
                # Look for explicit field mentions in the text
                lower = raw.lower()
                found_fields = [f for f in canonical if f.lower() in lower and f != "customer_id"]  # heuristic
                # Prefer field names that appear with positive language
                for f in canonical:
                    # Look for patterns like "matches X", "is X", "field X"
                    if f.lower() in lower:
                        # Pick the first canonical field mentioned, unless multiple — then we abstain
                        if found_fields and found_fields[0] == f:
                            master = f
                            conf = 0.7  # conservative since we inferred
                            break
                if not master:
                    return None
            # Validate
            if master.upper() == "NONE" or master == "":
                return None
            if master not in canonical:
                return None
            return {
                "master_field": master,
                "confidence":   conf,
                "reason":       parsed.get("reason", ""),
                "source":       "llm",
            }
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            return None
