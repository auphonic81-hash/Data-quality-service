"""PDF data extraction.

Two strategies:
  1. Text PDFs — digital invoices, exported reports. Use pdfplumber.
     Extracts tables natively. Near-perfect accuracy.
  2. Scanned/image PDFs — phone photos of paper documents. Use pytesseract OCR.
     Lower accuracy. Best for clean scans, struggles with handwriting.

Returns a pandas DataFrame ready to feed into the existing pipeline
(analyze, remediate, dedup against other files).
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


class PDFExtractor:
    """Extract structured data from PDFs."""

    def extract(self, pdf_path: str | Path) -> dict[str, Any]:
        """Pick the best strategy automatically and return extracted data.

        Returns:
          {
            "strategy": "text" | "ocr",
            "tables": [DataFrame, ...],   # tables found (text PDFs)
            "text": str,                  # raw text fallback
            "key_value": {...},           # k-v pairs parsed from text
            "ocr_confidence": float | None,
            "page_count": int,
            "warnings": [str, ...],
          }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return self._error(f"File not found: {pdf_path}")

        warnings: list[str] = []

        # ── Step 1: try text-based extraction first
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                tables: list[pd.DataFrame] = []
                text_parts: list[str] = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text_parts.append(page_text)
                    for tbl in page.extract_tables() or []:
                        df = self._table_to_dataframe(tbl)
                        if df is not None and not df.empty:
                            tables.append(df)
                full_text = "\n".join(text_parts).strip()

                if tables or len(full_text) > 50:
                    # Text-PDF path worked
                    return {
                        "strategy":       "text",
                        "tables":         tables,
                        "text":           full_text,
                        "key_value":      self._extract_key_values(full_text),
                        "ocr_confidence": None,
                        "page_count":     page_count,
                        "warnings":       warnings,
                    }
                warnings.append("PDF contains no extractable text — falling back to OCR")
        except Exception as exc:
            warnings.append(f"Text extraction failed ({type(exc).__name__}) — falling back to OCR")

        # ── Step 2: OCR fallback
        try:
            images = convert_from_path(str(pdf_path), dpi=300)
            ocr_texts: list[str] = []
            confidences: list[float] = []
            for img in images:
                data = pytesseract.image_to_data(
                    img, lang="eng", output_type=pytesseract.Output.DICT,
                )
                page_text = " ".join(t for t in data["text"] if t.strip())
                ocr_texts.append(page_text)
                page_confs = [int(c) for c in data["conf"] if c not in ("-1", -1)]
                if page_confs:
                    confidences.append(sum(page_confs) / len(page_confs))

            full_text = "\n".join(ocr_texts).strip()
            avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

            if avg_conf < 60:
                warnings.append(f"Low OCR confidence ({avg_conf}%) — extracted data may have errors.")

            tables = self._tables_from_ocr_text(full_text)

            return {
                "strategy":       "ocr",
                "tables":         tables,
                "text":           full_text,
                "key_value":      self._extract_key_values(full_text),
                "ocr_confidence": avg_conf,
                "page_count":     len(images),
                "warnings":       warnings,
            }
        except Exception as exc:
            return self._error(f"OCR failed: {exc}", warnings)

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _table_to_dataframe(raw: list[list[str]]) -> pd.DataFrame | None:
        """Convert a pdfplumber table (list-of-lists) to a DataFrame."""
        if not raw or len(raw) < 2:
            return None
        header = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(raw[0])]
        rows = []
        for row in raw[1:]:
            if not any(cell and str(cell).strip() for cell in row):
                continue
            rows.append([str(c).strip() if c else "" for c in row])
        if not rows:
            return None
        # Pad/truncate rows to header length
        rows = [r + [""] * (len(header) - len(r)) if len(r) < len(header) else r[:len(header)] for r in rows]
        return pd.DataFrame(rows, columns=header)

    @staticmethod
    def _extract_key_values(text: str) -> dict[str, str]:
        """Pull common invoice / bill fields out of free text using regex."""
        if not text:
            return {}
        patterns = {
            "invoice_no":   r"(?:invoice|bill|receipt)\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Za-z0-9\-_/]+)",
            "invoice_date": r"(?:invoice|bill|issue)\s*date\s*[:\-]?\s*([0-9]{1,4}[/\-.][0-9]{1,2}[/\-.][0-9]{1,4})",
            "due_date":     r"due\s*date\s*[:\-]?\s*([0-9]{1,4}[/\-.][0-9]{1,2}[/\-.][0-9]{1,4})",
            "total":        r"(?:total|amount|grand\s*total|balance\s*due)\s*[:\-]?\s*\$?([0-9,]+\.?[0-9]*)",
            "customer":     r"(?:bill\s*to|customer|client)\s*[:\-]?\s*([^\n\r]+?)(?=\s*(?:from|vendor|supplier|invoice|date|sku|description|$)|\n|\r)",
            "vendor":       r"(?:from|vendor|supplier)\s*[:\-]?\s*([^\n\r]+?)(?=\s*(?:sku|description|qty|invoice|date|$)|\n|\r)",
        }
        result: dict[str, str] = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip().rstrip(",.")
        return result

    @staticmethod
    def _tables_from_ocr_text(text: str) -> list[pd.DataFrame]:
        """Best-effort attempt to find tabular data in OCR text.

        Looks for lines with 3+ whitespace-separated tokens that appear
        to form columns. Returns at most one DataFrame.
        """
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        # Heuristic: rows with 3+ words separated by 2+ spaces look like columns
        candidate_rows = []
        for line in lines:
            parts = re.split(r"\s{2,}", line)
            if len(parts) >= 3:
                candidate_rows.append(parts)
        if len(candidate_rows) < 2:
            return []
        max_cols = max(len(r) for r in candidate_rows)
        normalized = [r + [""] * (max_cols - len(r)) for r in candidate_rows]
        header = normalized[0]
        # If header looks numeric, treat all rows as data
        if all(re.match(r"^\d", h or "") for h in header if h):
            df = pd.DataFrame(normalized, columns=[f"col_{i}" for i in range(max_cols)])
        else:
            df = pd.DataFrame(normalized[1:], columns=header)
        return [df]

    @staticmethod
    def _error(msg: str, warnings: list[str] | None = None) -> dict[str, Any]:
        return {
            "strategy":       None,
            "tables":         [],
            "text":           "",
            "key_value":      {},
            "ocr_confidence": None,
            "page_count":     0,
            "warnings":       (warnings or []) + [msg],
            "error":          msg,
        }
