"""Configuration for Data Quality Service."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR = BASE_DIR / "cache"

UPLOADS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:latest")

PORT = int(os.getenv("PORT", "5002"))
HOST = os.getenv("HOST", "0.0.0.0")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

MAX_UPLOAD_SIZE_MB = 100
SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"}

PROFILE_MINIMAL = True
PROFILE_SAMPLE_SIZE = 10000
