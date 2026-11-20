"""Configuration and credential loading for Thai Government OCR Benchmark."""

from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://opend.data.go.th/get-ckan"
HF_REPO_ID = "narawit16/thai-govern-test-ocr"

DATA_DIR = Path("data")
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
PAGES_DIR = DATA_DIR / "pages"
HF_DATASET_DIR = DATA_DIR / "hf_dataset"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
PAGE_INDEX_PATH = DATA_DIR / "page_index.jsonl"

GROUPS = [
    "politic",
    "economy",
    "energy",
    "map",
    "law",
    "government",
    "social",
    "public-health",
    "logistic",
    "research",
    "doi",
    "agriculture",
    "travel",
    "education",
    "culture",
]

ALLOWED_LICENSES = {
    "Open Data Common",
    "Creative Commons Attributions",
}

MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB
MIN_PDF_BYTES = 2 * 1024          # 2 KB
TARGET_PER_GROUP = 12             # 12 * 15 = ~180 PDFs
MAX_PER_ORG = 2
MAX_PAGES_PER_PDF = 4
RENDER_DPI = 200
REQUEST_TIMEOUT = 25

IMAGE_COVERAGE_THRESHOLD = 0.50
TARGET_TOTAL_PAGES = 600
SCANNED_RATIO = 0.75
MAX_SCAN_PAGES_PER_PDF = 30


@dataclass(frozen=True)
class Credentials:
    datagoth_api_key: str
    hf_token: str


def load_credentials(env_path: Path | str = ".env") -> Credentials:
    """Load data.go.th API key and Hugging Face token from .env."""
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing credentials file: {path}")

    datagoth_key = ""
    hf_token = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k_clean = k.strip().lower()
            v_clean = v.strip().strip('"').strip("'")
            if k_clean in {"thai-ocr", "hf_token", "huggingface_token"}:
                hf_token = v_clean
            elif k_clean in {"datagoth_api_key", "data_go_th_key", "api_key"}:
                datagoth_key = v_clean
        elif not datagoth_key:
            datagoth_key = line

    if not datagoth_key:
        raise ValueError("data.go.th API key not found in .env")
    if not hf_token:
        raise ValueError("Hugging Face token (thai-ocr) not found in .env")

    return Credentials(datagoth_api_key=datagoth_key, hf_token=hf_token)
