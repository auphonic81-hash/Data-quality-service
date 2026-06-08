"""Schema inference module.

Combines pandas type inference with LLM-driven schema design.
Outputs production-ready CREATE TABLE statements with constraints.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import requests


class SchemaInferencer:
    """Infers a complete database schema from a DataFrame."""

    SQL_TYPE_MAP = {
        "int64": "INTEGER",
        "int32": "INTEGER",
        "float64": "REAL",
        "float32": "REAL",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
        "object": "TEXT",
        "category": "TEXT",
    }

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 ollama_model: str = "qwen3.5:latest"):
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model

    def infer(self, df: pd.DataFrame, table_name: str = "data") -> dict[str, Any]:
        """Infer schema with optional LLM enhancement.

        Returns:
            {
              "table_name": str,
              "columns": [{name, sql_type, nullable, unique, primary_key, ...}],
              "create_statement": str,
              "llm_suggestions": dict | None
            }
        """
        if df.empty:
            raise ValueError("Cannot infer schema from empty DataFrame")

        columns = self._analyze_columns(df)
        create_sql = self._build_create_statement(table_name, columns)
        llm_suggestions = self._llm_enhance(df, columns, table_name)

        return {
            "table_name": table_name,
            "columns": columns,
            "create_statement": create_sql,
            "llm_suggestions": llm_suggestions,
        }

    # ─── Pandas-based analysis ─────────────────────────────────────────────

    def _analyze_columns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Analyze each column to determine SQL type and constraints."""
        columns: list[dict[str, Any]] = []
        row_count = len(df)

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()

            sql_type = self._infer_sql_type(series)
            null_count = int(series.isna().sum())
            unique_count = int(non_null.nunique()) if len(non_null) else 0

            is_unique = unique_count == len(non_null) and len(non_null) > 0
            is_required = null_count == 0
            is_pk_candidate = is_unique and is_required and len(non_null) == row_count

            col_info = {
                "name": col,
                "sql_type": sql_type,
                "nullable": not is_required,
                "unique": is_unique,
                "primary_key_candidate": is_pk_candidate,
                "null_count": null_count,
                "unique_count": unique_count,
                "sample_values": [str(v) for v in non_null.head(3).tolist()],
            }

            # Add type-specific constraints
            if sql_type in {"INTEGER", "REAL"} and len(non_null):
                try:
                    col_info["min_value"] = float(non_null.min())
                    col_info["max_value"] = float(non_null.max())
                except (TypeError, ValueError):
                    pass
            elif sql_type == "TEXT" and len(non_null):
                col_info["max_length"] = int(non_null.astype(str).str.len().max())

            columns.append(col_info)

        return columns

    def _infer_sql_type(self, series: pd.Series) -> str:
        """Map pandas dtype to SQL type with content inspection."""
        dtype_str = str(series.dtype)

        if dtype_str in self.SQL_TYPE_MAP:
            base_type = self.SQL_TYPE_MAP[dtype_str]
        else:
            base_type = "TEXT"

        # If TEXT but values are mostly numeric, suggest converting
        if base_type == "TEXT":
            non_null = series.dropna()
            if len(non_null) > 0:
                numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().sum() / len(non_null)
                if numeric_ratio > 0.95:
                    # Could be cleanable to numeric
                    if (pd.to_numeric(non_null, errors="coerce").dropna() % 1 == 0).all():
                        return "INTEGER"
                    return "REAL"

        return base_type

    def _build_create_statement(self, table_name: str, columns: list[dict[str, Any]]) -> str:
        """Build a SQL CREATE TABLE statement."""
        safe_table = re.sub(r"[^A-Za-z0-9_]", "_", table_name)
        column_defs = []

        primary_key_set = False
        for col in columns:
            parts = [f'"{col["name"]}"', col["sql_type"]]

            if col.get("primary_key_candidate") and not primary_key_set:
                parts.append("PRIMARY KEY")
                primary_key_set = True
            else:
                if not col["nullable"]:
                    parts.append("NOT NULL")
                if col["unique"] and not col.get("primary_key_candidate"):
                    parts.append("UNIQUE")

            column_defs.append("  " + " ".join(parts))

        return f'CREATE TABLE "{safe_table}" (\n' + ",\n".join(column_defs) + "\n);"

    # ─── LLM enhancement ───────────────────────────────────────────────────

    def _llm_enhance(
        self, df: pd.DataFrame, columns: list[dict[str, Any]], table_name: str
    ) -> dict[str, Any] | None:
        """Ask the LLM for higher-level schema improvements.

        Returns suggestions like:
        - Better column types
        - Suggested indexes
        - Detected entity (this looks like a 'Contacts' table)
        - Relationship hints
        """
        try:
            sample = df.head(5).to_dict(orient="records")
            prompt = self._build_llm_prompt(table_name, columns, sample)
            response = self._call_ollama(prompt)
            return self._parse_llm_response(response)
        except Exception as exc:
            return {"error": f"LLM enhancement skipped: {exc}"}

    def _build_llm_prompt(
        self, table_name: str, columns: list[dict[str, Any]], sample: list[dict]
    ) -> str:
        compact_columns = [
            {
                "name": c["name"],
                "type": c["sql_type"],
                "nullable": c["nullable"],
                "unique": c["unique"],
                "sample": c["sample_values"][:2],
            }
            for c in columns
        ]
        return (
            "You are a senior database architect. Analyze this table schema and suggest improvements.\n\n"
            f"Table name: {table_name}\n"
            f"Columns: {json.dumps(compact_columns, ensure_ascii=False)}\n"
            f"Sample rows: {json.dumps(sample, ensure_ascii=False)[:1000]}\n\n"
            "Return ONLY a JSON object with these keys:\n"
            "  - detected_entity: what this table represents (e.g. 'customer contacts')\n"
            "  - suggested_indexes: list of column names that should be indexed\n"
            "  - type_corrections: dict mapping column name → better SQL type\n"
            "  - relationship_hints: list of strings describing potential foreign keys\n"
            "  - notes: brief observations about the schema\n\n"
            "Return JSON only, no commentary."
        )

    def _call_ollama(self, prompt: str) -> str:
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 800},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    @staticmethod
    def _parse_llm_response(content: str) -> dict[str, Any]:
        """Parse LLM JSON response, with fallback regex extraction."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if match:
                return json.loads(match.group(0))
            return {"raw": content}