#!/usr/bin/env python3
"""Reset the platform to a clean state for a fresh demo run.

Steps (in order):
  1. Restore files from aside/ and duplicates/ back to their original paths
     (reads sidecar files to find where they came from)
  2. Export landing_file_* tables to CSV files in ~/demo_sources/local_folder/
     ONLY for files that don't already exist there. This ensures we have
     working source files even if the originals were deleted after successful
     ingestion (delete-on-success behavior).
  3. Wipe all rows from landing_*, stg_*, master_* tables, plus audit tables
     (file_hashes, file_quarantine, ingestion_log, datasets, analyses).
     Schemas are preserved.
  4. Empty the aside/ and duplicates/ directories.
  5. Verify demo source files are present.

SAFETY: Requires --confirm to actually perform changes. Without it, dry-run.

Usage:
  python ~/reset_platform.py             # Dry run
  python ~/reset_platform.py --confirm   # Actually reset
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sqlite3
import sys
from pathlib import Path

# --- Paths ---
CATALOG_DB      = Path.home() / "data-quality-service" / "reports" / "catalog.sqlite3"
ASIDE_DIR       = Path.home() / "data-quality-service" / "aside"
DUPLICATES_DIR  = Path.home() / "data-quality-service" / "duplicates"
LOCAL_FOLDER    = Path.home() / "demo_sources" / "local_folder"

# --- Landing tables to export to CSV in local_folder/ ---
# Only exports if target doesn't already exist (won't overwrite restored files)
LANDING_TO_CSV = {
    "landing_file_customers":                 "customers.csv",
    "landing_file_customers_branch_export":   "customers_branch_export.csv",
    "landing_file_employees":                 "employees.csv",
    "landing_file_invoices":                  "invoices.csv",
    "landing_file_orders":                    "orders.csv",
    "landing_file_products":                  "products.csv",
    "landing_file_vendors_new":               "vendors_new.csv",
    "landing_file_vendors_master":            "vendors_master.csv",
    "landing_file_vendors_old":               "vendors_old.csv",
}

# --- Demo source files that should exist after reset ---
EXPECTED_SOURCES = [
    LOCAL_FOLDER / "customers.csv",
    LOCAL_FOLDER / "customers_branch_export.csv",
    LOCAL_FOLDER / "orders.csv",
    LOCAL_FOLDER / "products.csv",
    LOCAL_FOLDER / "vendors_new.csv",
    LOCAL_FOLDER / "invoices.csv",
    LOCAL_FOLDER / "employees.csv",
    Path.home() / "demo_sources" / "sharepoint_finance" / "customers_finance_copy.xlsx",
    Path.home() / "demo_sources" / "sharepoint_finance" / "new_finance_hires_q4.csv",
]


# ============================================================
# STEP 1 - Restore quarantined files back to their source dirs
# ============================================================
def _parse_original_path(sidecar_path: Path) -> str | None:
    try:
        content = sidecar_path.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            if line.startswith("Original path:"):
                return line.split(":", 1)[1].strip()
    except Exception as e:
        print(f"  [warn] could not read {sidecar_path.name}: {e}")
    return None


def restore_files(source_dir: Path, sidecar_suffix: str, dry_run: bool) -> int:
    if not source_dir.exists():
        return 0
    restored = 0
    for sidecar in source_dir.glob(f"*{sidecar_suffix}"):
        actual_file = source_dir / sidecar.name[: -len(sidecar_suffix)]
        if not actual_file.exists():
            continue
        original_path = _parse_original_path(sidecar)
        if not original_path:
            continue
        target = Path(original_path)
        if target.exists():
            print(f"  [skip] target exists: {target}")
            continue
        if dry_run:
            print(f"  [would restore] {actual_file.name} -> {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(actual_file), str(target))
            sidecar.unlink()
            print(f"  [restored] {target}")
        restored += 1
    return restored


# ============================================================
# STEP 2 - Export landing tables to CSV
# ============================================================
def export_landing_tables(dry_run: bool) -> int:
    if not CATALOG_DB.exists():
        return 0
    exported = 0
    LOCAL_FOLDER.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(CATALOG_DB), timeout=30) as con:
        for table, csv_name in LANDING_TO_CSV.items():
            target = LOCAL_FOLDER / csv_name
            if target.exists():
                print(f"  [skip] {csv_name} already exists")
                continue
            has = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not has:
                print(f"  [skip] {table} not in DB")
                continue
            cols_info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
            cols = [c[1] for c in cols_info if not c[1].startswith("_")]
            if not cols:
                continue
            count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if count == 0:
                continue
            if dry_run:
                print(f"  [would export] {table} ({count} rows) -> {target}")
            else:
                select_cols = ", ".join(f'"{c}"' for c in cols)
                rows = con.execute(f'SELECT {select_cols} FROM "{table}"').fetchall()
                with open(target, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(cols)
                    w.writerows(rows)
                print(f"  [exported] {csv_name}: {count} rows")
            exported += 1
    return exported


# ============================================================
# STEP 3 - Wipe DB tables
# ============================================================
def wipe_tables(dry_run: bool) -> int:
    if not CATALOG_DB.exists():
        return 0
    total = 0
    with sqlite3.connect(str(CATALOG_DB), timeout=30) as con:
        cur = con.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
              AND (name LIKE 'landing_%' OR name LIKE 'stg_%' OR name LIKE 'master_%')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        pattern_tables = [row[0] for row in cur.fetchall()]
        explicit_tables = ["file_hashes", "file_quarantine", "ingestion_log", "datasets", "analyses"]
        all_tables = pattern_tables + [
            t for t in explicit_tables
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
        ]
        print(f"  Tables to wipe: {len(all_tables)}")
        for table in all_tables:
            try:
                count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            except Exception as e:
                print(f"  [skip] {table}: {e}")
                continue
            if count == 0:
                continue
            if dry_run:
                print(f"  [would delete] {table}: {count} rows")
            else:
                con.execute(f'DELETE FROM "{table}"')
                print(f"  [wiped] {table}: {count} rows")
            total += count
        if not dry_run:
            con.commit()  # skip VACUUM to avoid transaction error
    return total


# ============================================================
# STEP 4 - Empty dirs
# ============================================================
def empty_dir(target: Path, dry_run: bool) -> int:
    if not target.exists():
        return 0
    count = 0
    for item in target.iterdir():
        if item.is_file():
            if dry_run:
                print(f"  [would remove] {item.name}")
            else:
                item.unlink()
                print(f"  [removed] {item.name}")
            count += 1
    return count


# ============================================================
# STEP 5 - Verify sources
# ============================================================
def verify_sources() -> tuple[list[Path], list[Path]]:
    return (
        [p for p in EXPECTED_SOURCES if p.exists()],
        [p for p in EXPECTED_SOURCES if not p.exists()],
    )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true", help="Actually perform the reset")
    args = parser.parse_args()
    dry_run = not args.confirm
    banner = "DRY RUN - nothing will change" if dry_run else "LIVE - changes will be made"

    print("=" * 60)
    print(f"PLATFORM RESET  ({banner})")
    print("=" * 60)

    print("\n--- STEP 1: Restore quarantined files ---")
    restored_a = restore_files(ASIDE_DIR, ".reason.txt", dry_run)
    restored_d = restore_files(DUPLICATES_DIR, ".duplicate_of.txt", dry_run)
    print(f"  From aside/: {restored_a}, from duplicates/: {restored_d}")

    print("\n--- STEP 2: Export landing tables to CSV (only if target missing) ---")
    exported = export_landing_tables(dry_run)
    print(f"  CSVs exported: {exported}")

    print("\n--- STEP 3: Wipe DB tables ---")
    rows = wipe_tables(dry_run)
    print(f"  Total rows deleted: {rows}")

    print("\n--- STEP 4: Empty aside/ and duplicates/ ---")
    removed_a = empty_dir(ASIDE_DIR, dry_run)
    removed_d = empty_dir(DUPLICATES_DIR, dry_run)
    print(f"  Removed from aside/: {removed_a}, from duplicates/: {removed_d}")

    print("\n--- STEP 5: Verify demo source files ---")
    present, missing = verify_sources()
    print(f"  Present: {len(present)}/{len(EXPECTED_SOURCES)}")
    if missing:
        print(f"  MISSING:")
        for p in missing:
            print(f"    - {p}")
        print(f"\n  Run: python ~/populate_sharepoint.py  (for SharePoint files)")

    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE. Re-run with --confirm to actually reset.")
    else:
        print("RESET COMPLETE.")
        print()
        print("Next steps:")
        print("  1. Restart Flask:")
        print("     fuser -k 5002/tcp; sleep 2")
        print("     cd ~/data-quality-service && python app.py > /tmp/flask.log 2>&1 & disown")
        print("     sleep 5")
        print("  2. python ~/populate_sharepoint.py   # regen SharePoint files")
        print("  3. Verify empty state:")
        print("     sqlite3 ~/data-quality-service/reports/catalog.sqlite3 \\")
        print('       "SELECT COUNT(*) FROM master_customers;"')
        print("  4. Run full ingestion via UI or curl to populate for demo")
    print("=" * 60)


if __name__ == "__main__":
    main()