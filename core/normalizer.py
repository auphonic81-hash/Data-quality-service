"""Schema Normalization Engine.

Takes a denormalized (flat) table and produces a properly normalized schema
in 3NF by detecting functional dependencies.

Algorithm:
1. For each column pair, compute uniqueness ratio of one given the other
   (a "functional dependency" means: knowing A determines B)
2. Cluster columns that form mutual dependency groups → these are entities
3. The remaining columns + foreign keys to entities → main fact table
4. Generate CREATE TABLE statements with PRIMARY KEY + FOREIGN KEY constraints
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import pandas as pd


class SchemaNormalizer:
    """Detects denormalized data and produces a normalized schema."""

    # How "tight" the functional dependency must be to consider columns linked.
    # 1.0 means: every distinct value of A maps to exactly one value of B.
    FD_THRESHOLD = 0.98

    # If a column has this fraction of unique values, it's the primary key of the fact table.
    PK_UNIQUENESS = 0.98

    MIN_ROWS_FOR_NORMALIZATION = 20
    FD_SAMPLE_SIZE = 5000

    def normalize(self, df: pd.DataFrame, source_table_name: str = "data") -> dict[str, Any]:
        """Run the full normalization pipeline."""
        if df.empty:
            raise ValueError("Cannot normalize an empty DataFrame")

        # Too few rows → can't reliably detect functional dependencies
        if len(df) < self.MIN_ROWS_FOR_NORMALIZATION:
            return {
                "source_table": source_table_name,
                "is_denormalized": False,
                "entity_count": 0,
                "entities": [],
                "fact_table_columns": list(df.columns),
                "create_statements": {},
                "skipped_reason": f"Dataset too small ({len(df)} rows). Need >= {self.MIN_ROWS_FOR_NORMALIZATION}.",
            }

        # Step 1: find functional dependencies between every pair of columns
        # Sample large datasets to keep this O(cols^2) step tractable
        df_for_fd = df if len(df) <= self.FD_SAMPLE_SIZE else df.sample(
            n=self.FD_SAMPLE_SIZE, random_state=42
        )
        fd_matrix = self._compute_fd_matrix(df_for_fd)

        # Step 2: cluster columns into entity groups based on mutual FDs
        entity_groups = self._cluster_into_entities(df, fd_matrix)

        # Step 3: figure out the fact table (columns not absorbed into entities)
        all_entity_cols = {col for group in entity_groups for col in group["columns"]}
        fact_columns = [c for c in df.columns if c not in all_entity_cols]

        # Step 4: name the entities
        for group in entity_groups:
            group["entity_name"] = self._suggest_entity_name(group["columns"])
            group["row_count"] = int(df[group["columns"]].drop_duplicates().shape[0])
            group["sample"] = (
                df[group["columns"]].drop_duplicates().head(3).fillna("").astype(str).to_dict(orient="records")
            )

        # Step 5: generate DDL
        ddl = self._generate_ddl(
            df,
            entity_groups,
            fact_columns,
            fact_table_name=source_table_name + "_facts",
        )

        return {
            "source_table": source_table_name,
            "is_denormalized": len(entity_groups) > 0,
            "entity_count": len(entity_groups),
            "entities": entity_groups,
            "fact_table_columns": fact_columns,
            "create_statements": ddl,
        }

    # ─── Step 1: functional dependencies ──────────────────────────────────

    def _compute_fd_matrix(self, df: pd.DataFrame) -> dict[tuple[str, str], float]:
        """For each (A, B), compute how often A → B holds across the dataset."""
        fd: dict[tuple[str, str], float] = {}
        columns = list(df.columns)

        for col_a in columns:
            # group by A, see how many unique B values per group
            grouped = df.groupby(col_a, dropna=False)
            for col_b in columns:
                if col_a == col_b:
                    continue
                # unique values of B per group of A
                unique_per_group = grouped[col_b].nunique(dropna=False)
                # FD holds when every group has exactly 1 unique B value
                holds = (unique_per_group <= 1).sum()
                total = len(unique_per_group)
                fd[(col_a, col_b)] = holds / total if total else 0.0

        return fd

    # ─── Step 2: cluster columns into entity groups ───────────────────────

    def _cluster_into_entities(
        self, df: pd.DataFrame, fd_matrix: dict[tuple[str, str], float]
    ) -> list[dict[str, Any]]:
        """Find groups of columns that describe the same entity.

        Correct algorithm:
          1. Find "anchor" columns — columns whose distinct values are far fewer
             than the row count (these represent repeated entities).
          2. For each anchor, gather all columns it strictly determines.
          3. Each anchor + its dependents = one entity.
        """
        columns = list(df.columns)
        n_rows = len(df)

        # Find anchors: low-cardinality columns that determine others
        anchor_candidates: list[tuple[str, int]] = []
        for col in columns:
            distinct = df[col].nunique(dropna=False)
            # An anchor has far fewer distinct values than rows
            if distinct < n_rows * 0.5 and distinct >= 2:
                anchor_candidates.append((col, distinct))

        # Sort anchors so smaller-cardinality columns are tried first
        anchor_candidates.sort(key=lambda x: x[1])

        used_columns: set[str] = set()
        groups: list[list[str]] = []

        for anchor, _ in anchor_candidates:
            if anchor in used_columns:
                continue
            # Find all columns this anchor strictly determines (and that aren't already taken)
            dependents = [anchor]
            for other in columns:
                if other == anchor or other in used_columns:
                    continue
                if fd_matrix.get((anchor, other), 0) >= self.FD_THRESHOLD:
                    dependents.append(other)

            # Only form an entity if the anchor pulls in at least 1 other column
            if len(dependents) >= 2:
                # Verify the group is genuinely repeated (not nearly unique)
                distinct_rows = df[dependents].drop_duplicates().shape[0]
                if distinct_rows < n_rows * 0.5:
                    groups.append(dependents)
                    used_columns.update(dependents)

        # Build the result objects
        real_entities: list[dict[str, Any]] = []
        for group in groups:
            distinct_rows = df[group].drop_duplicates().shape[0]
            real_entities.append({
                "columns": sorted(group),
                "distinct_rows": int(distinct_rows),
                "reduction_ratio": round(1 - distinct_rows / n_rows, 3),
            })

        real_entities.sort(key=lambda g: -g["reduction_ratio"])
        return real_entities

    # ─── Step 3: name entities ────────────────────────────────────────────

    def _suggest_entity_name(self, columns: list[str]) -> str:
        """Pick a sensible table name from the column prefixes."""
        # Find common prefix (e.g. owner_name + owner_email → "owner")
        prefixes: list[str] = []
        for col in columns:
            tokens = re.split(r"[_\W]+", col.lower())
            if tokens:
                prefixes.append(tokens[0])

        if prefixes:
            from collections import Counter
            most_common = Counter(prefixes).most_common(1)[0]
            if most_common[1] >= len(columns) / 2:
                # most columns share this prefix
                name = most_common[0]
                # pluralize lightly
                if not name.endswith("s"):
                    name = name + "s"
                return name.capitalize()

        # Fallback: join first parts of column names
        return "Entity_" + "_".join(c[:4] for c in columns[:2])

    # ─── Step 4: generate DDL ─────────────────────────────────────────────

    def _generate_ddl(
        self,
        df: pd.DataFrame,
        entity_groups: list[dict[str, Any]],
        fact_columns: list[str],
        fact_table_name: str,
    ) -> dict[str, str]:
        """Generate CREATE TABLE statements for each entity + the fact table."""
        statements: dict[str, str] = {}

        # Entity tables
        for group in entity_groups:
            table_name = group["entity_name"]
            lines = [f'  "{table_name.lower()}_id" INTEGER PRIMARY KEY']
            for col in group["columns"]:
                sql_type = self._sql_type(df[col], col)
                lines.append(f'  "{col}" {sql_type}')
            statements[table_name] = (
                f'CREATE TABLE "{table_name}" (\n'
                + ",\n".join(lines)
                + "\n);"
            )

        # Fact table — pick exactly ONE primary key (highest uniqueness, ideally an "id" column)
        fact_lines: list[str] = []
        pk_col = self._pick_primary_key(df, fact_columns)
        for col in fact_columns:
            sql_type = self._sql_type(df[col], col)
            if col == pk_col:
                fact_lines.append(f'  "{col}" {sql_type} PRIMARY KEY')
            else:
                fact_lines.append(f'  "{col}" {sql_type}')

        # Foreign key references to each entity
        fk_lines: list[str] = []
        for group in entity_groups:
            entity_name = group["entity_name"]
            fk_col = f"{entity_name.lower().rstrip('s')}_id"
            fact_lines.append(f'  "{fk_col}" INTEGER')
            fk_lines.append(
                f'  FOREIGN KEY ("{fk_col}") REFERENCES "{entity_name}" ("{entity_name.lower()}_id")'
            )

        all_lines = fact_lines + fk_lines
        statements[fact_table_name] = (
            f'CREATE TABLE "{fact_table_name}" (\n'
            + ",\n".join(all_lines)
            + "\n);"
        )

        return statements

    @staticmethod
    def _pick_primary_key(df: pd.DataFrame, candidate_cols: list[str]) -> str | None:
        """Pick exactly one PK column. Prefer columns named *_id with full uniqueness."""
        scored: list[tuple[str, float, int]] = []
        for col in candidate_cols:
            uniqueness = df[col].nunique() / len(df) if len(df) else 0
            # Bonus if column name suggests it's an identifier
            id_bonus = 1 if re.search(r"(^id$|_id$|^id_)", col.lower()) else 0
            scored.append((col, uniqueness, id_bonus))
        # Highest uniqueness wins, id-named columns break ties
        scored.sort(key=lambda x: (-x[1], -x[2]))
        if scored and scored[0][1] >= 0.98:
            return scored[0][0]
        return None

    @staticmethod
    def _sql_type(series: pd.Series, col_name: str = "") -> str:
        """Map a pandas series to a SQL type. Column name overrides numeric detection
        for known string-typed fields (phone, postal codes, etc.)."""
        name = (col_name or str(series.name or "")).lower()
        # Identifier-like strings must stay TEXT even if they look numeric
        if any(token in name for token in ("phone", "mobile", "fax", "zip", "postal", "code")):
            return "TEXT"

        dtype = str(series.dtype)
        if "int" in dtype:
            return "INTEGER"
        if "float" in dtype:
            return "REAL"
        if "bool" in dtype:
            return "BOOLEAN"
        if "datetime" in dtype:
            return "TIMESTAMP"

        non_null = series.dropna()
        if len(non_null) == 0:
            return "TEXT"
        numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().sum() / len(non_null)
        if numeric_ratio > 0.95:
            if (pd.to_numeric(non_null, errors="coerce").dropna() % 1 == 0).all():
                return "INTEGER"
            return "REAL"
        return "TEXT"

    def materialize(
        self, df: pd.DataFrame, entities: list[dict[str, Any]],
        fact_columns: list[str], source_table_name: str = "data"
    ) -> dict[str, pd.DataFrame]:
        """Physically split a flat DataFrame into normalized tables.

        Args:
            df: The original denormalized DataFrame.
            entities: The entity groups returned by `normalize()`.
            fact_columns: Columns that belong to the fact table (not absorbed
                into any entity).
            source_table_name: Used to name the fact table.

        Returns:
            dict mapping table_name → DataFrame
              - Each entity table has a surrogate primary key + its columns
              - The fact table has its own columns + foreign keys to each entity
        """
        tables: dict[str, pd.DataFrame] = {}

        # Working copy of the fact table starts as full df (we'll prune later)
        fact_df = df[fact_columns].copy()

        for entity in entities:
            entity_name = entity["entity_name"]
            entity_cols = entity["columns"]
            pk_col = f"{entity_name.lower()}_id"
            fk_col = f"{entity_name.lower().rstrip('s')}_id"

            # Deduplicate to get unique rows
            entity_df = (
                df[entity_cols]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            entity_df.insert(0, pk_col, range(1, len(entity_df) + 1))
            tables[entity_name] = entity_df

            # Build lookup: tuple-of-entity-values → surrogate id
            lookup_df = entity_df.set_index(entity_cols)[pk_col]
            # Map each row of df to the matching surrogate id
            mapped_ids = df.set_index(entity_cols).index.map(lookup_df.to_dict())
            fact_df[fk_col] = list(mapped_ids)

        fact_table_name = f"{source_table_name}_facts"
        tables[fact_table_name] = fact_df.reset_index(drop=True)
        return tables

