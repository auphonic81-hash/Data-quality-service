# Unified Data Platform — Lightweight MDM

A Master Data Management platform built as a mid-market alternative to Informatica MDM Hub. Reproduces the three-layer landing → staging → master pipeline in a single Python/Flask codebase.

## What it does

Ingests data from multiple sources (files, REST APIs, SQLite databases, PDFs), reconciles them against 14+ configured business entities, and produces a golden master record with full lineage tracking.

## Key features

- **Multi-source ingestion** — CSV, XLSX, JSON, Parquet, REST APIs, SQLite, and PDFs (with OCR fallback for scanned images)
- **Content-hash deduplication** — Detects identical data across different file formats (Excel matches a CSV of the same rows)
- **PDF intelligence** — Extracts structured data from invoices using fuzzy vendor matching, enriches existing master records
- **Auto-entity onboarding** — LLM-powered classification and schema inference for new entity types not seen before, with guardrails against garbage input
- **Probabilistic linkage** — Splink (Fellegi-Sunter) integration for cross-source fuzzy matching
- **Full lineage** — Per-record, per-field provenance tracking with source attribution and timestamps
- **Steward workbench** — UI for reviewing match candidates with approve/reject persistence

## Architecture

Follows Informatica MDM Hub's reference architecture:

- **Landing layer**: raw source data preserved
- **Staging layer**: canonical schema mapping via configurable aliases
- **Master layer**: consolidated golden records with best-version-of-truth semantics

## Tech stack

- Python 3.11 / Flask 3.x
- SQLite for catalog and master storage
- Pandera for schema validation
- Splink for probabilistic record linkage
- Pytesseract + pdfplumber for PDF extraction
- Ollama (qwen3) for LLM-based schema proposals

## Screenshots

_(Add screenshots of the dashboard, master data drawer, and auto-onboard flow)_

## Setup

```bash
git clone <repo-url>
cd data-quality-service
pip install -r requirements.txt
python app.py
```

Open http://localhost:5002.

## Status

Working prototype. Currently demonstrating the core pipeline pattern. Production deployment (auth, multi-tenancy, hosted infrastructure) is on the roadmap.

## Author

Saleh — Computer Engineering background, based in UAE.
