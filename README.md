# Data Quality Service

An open-source enterprise-grade data quality engine. Profiles, validates, and remediates messy datasets automatically.

## Features

- **Multi-format ingestion** — CSV, Excel, JSON, Parquet, SQLite, TSV
- **Automated profiling** — Powered by ydata-profiling with custom Gulf-aware checks
- **Schema inference** — Pandas-based type detection + LLM-enhanced suggestions
- **Quality detection** — Duplicates, missing values, type inconsistencies, outliers
- **Smart remediation** — Phone (E.164), email validation, date parsing, name clustering
- **Safe fuzzy matching** — Token-level similarity prevents false merges
- **Side-by-side diff view** — See exactly what changed
- **Arabic + English** — Bilingual data support

## Stack

Python · Flask · pandas · ydata-profiling · rapidfuzz · dedupe · phonenumbers · dateparser · Ollama

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5002`
