"""Stage-scoped configuration and isolated credential loading for train_ocr."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://opend.data.go.th/get-ckan"
HF_REPO_ID = "narawit16/thai-govern-test-ocr"

DATA_DIR = Path(os.environ.get("TRAIN_OCR_DATA_DIR", "data"))
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
MIN_PDF_BYTES = 2 * 1024  # 2 KB
TARGET_PER_GROUP = 12  # 12 * 15 = ~180 PDFs
MAX_PER_ORG = 2
MAX_PAGES_PER_PDF = 4
RENDER_DPI = 200
REQUEST_TIMEOUT = 25

IMAGE_COVERAGE_THRESHOLD = 0.50
TARGET_TOTAL_PAGES = 600
SCANNED_RATIO = 0.75
MAX_SCAN_PAGES_PER_PDF = 30

# Saraban 6-Type Benchmark Configuration
SARABAN_DIR = DATA_DIR / "saraban"
SARABAN_RAW_DIR = SARABAN_DIR / "raw_pdfs"
SARABAN_PAGES_DIR = SARABAN_DIR / "pages"
SARABAN_HF_DIR = SARABAN_DIR / "hf_dataset"
SARABAN_MANIFEST_PATH = SARABAN_DIR / "manifest.jsonl"
SARABAN_PAGE_INDEX_PATH = SARABAN_DIR / "page_index.jsonl"

SARABAN_RENDER_DPI = 300
SARABAN_TARGET_PER_TYPE = 70
SARABAN_MAX_PAGES_PER_PDF = 3
SARABAN_SCANNED_RATIO = 0.75

SARABAN_TYPES = [
    "external_letter",
    "internal_memo",
    "stamped_letter",
    "order_regulation",
    "public_relations",
    "official_record",
]

PII_TITLE_BLOCKLIST = [
    "แต่งตั้งโยกย้าย",
    "เลื่อนเงินเดือน",
    "สวัสดิการ",
    "บำเหน็จ",
    "บำนาญ",
    "ผู้มีสิทธิรับเงิน",
    "รายชื่อผู้ผ่าน",
    "รายชื่อผู้มีสิทธิ",
    "พ้นจากราชการ",
    "เกษียณอายุ",
    "สอบแข่งขัน",
    "ชี้มูลความผิด",
    "บรรจุและแต่งตั้ง",
    "โอนย้าย",
    "ย้ายข้าราชการ",
    "รับสมัครบุคคล",
    "คัดเลือกบุคคล",
]

# Google Images Crawler & Screener Microservice Configuration
GOOGLE_CRAWL_DIR = DATA_DIR / "google_images"
GOOGLE_PREPROCESSED_DIR = GOOGLE_CRAWL_DIR / "preprocessed"
GOOGLE_HF_DIR = GOOGLE_CRAWL_DIR / "hf_dataset"
GOOGLE_BLOCKED_DOMAINS = (
    "lemon8-app.com",
    "instagram.com",
    "slidesharecdn.com",
    "facebook.com",
    "fbsbx.com",
    "fbcdn.net",
    "pinterest.com",
    "pinimg.com",
    "tiktok.com",
)
DEFAULT_OCR_SCREENER_URL = os.environ.get("OCR_SCREENER_URL", "")

# Golden Set (120 pages) & Ground Truth / Evaluation Configuration
GOLDEN_DIR = SARABAN_DIR / "golden_120"
GOLDEN_MANIFEST_PATH = GOLDEN_DIR / "golden_manifest.jsonl"
GOLDEN_DRAFTS_PATH = GOLDEN_DIR / "dual_drafts.jsonl"
GT_ANNOTATIONS_PATH = GOLDEN_DIR / "gt_annotations.jsonl"
EVAL_REPORT_MD_PATH = GOLDEN_DIR / "eval_report.md"
EVAL_REPORT_CSV_PATH = GOLDEN_DIR / "eval_report.csv"

# DLA Bulk Harvester Configuration
DLA_BULK_DIR = DATA_DIR / "dla_bulk"
DLA_BULK_DB_PATH = DLA_BULK_DIR / "catalog.sqlite"
DLA_BULK_FILES_DIR = DLA_BULK_DIR / "files"
DLA_BULK_MANIFEST_PATH = DLA_BULK_DIR / "manifest.jsonl"
DLA_BULK_PAGE_INDEX_PATH = DLA_BULK_DIR / "page_index.jsonl"
DLA_BULK_PAGES_DIR = DLA_BULK_DIR / "pages"
DLA_BULK_HF_DIR = DLA_BULK_DIR / "hf_dataset"
DLA_BULK_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB per file
DLA_BULK_MAX_TOTAL_BYTES = 1024 * 1024 * 1024  # 1 GB default total budget
DLA_BULK_DEFAULT_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".gif", ".bmp")

GEMINI_DEFAULT_MODEL = "gemini-2.5-pro"
IAPP_MARKDOWN_URL = "https://api.iapp.co.th/v3/store/ocr/document/markdown"
KKU_BASE_URL = "https://gen.ai.kku.ac.th/api/v1"
KKU_MODEL_A = "gemini-4-hydrogen"
KKU_MODEL_B = "gemma4-26b"


@dataclass(frozen=True)
class DataPaths:
    data_dir: Path = DATA_DIR
    raw_pdf_dir: Path = RAW_PDF_DIR
    pages_dir: Path = PAGES_DIR
    hf_dataset_dir: Path = HF_DATASET_DIR
    manifest_path: Path = MANIFEST_PATH
    page_index_path: Path = PAGE_INDEX_PATH
    saraban_dir: Path = SARABAN_DIR
    saraban_raw_dir: Path = SARABAN_RAW_DIR
    saraban_pages_dir: Path = SARABAN_PAGES_DIR
    saraban_hf_dir: Path = SARABAN_HF_DIR
    saraban_manifest_path: Path = SARABAN_MANIFEST_PATH
    saraban_page_index_path: Path = SARABAN_PAGE_INDEX_PATH
    google_crawl_dir: Path = GOOGLE_CRAWL_DIR


@dataclass(frozen=True)
class DataGoThSettings:
    base_url: str = BASE_URL
    groups: tuple[str, ...] = tuple(GROUPS)
    allowed_licenses: frozenset[str] = frozenset(ALLOWED_LICENSES)
    max_pdf_bytes: int = MAX_PDF_BYTES
    min_pdf_bytes: int = MIN_PDF_BYTES
    target_per_group: int = TARGET_PER_GROUP
    max_per_org: int = MAX_PER_ORG
    request_timeout: int = REQUEST_TIMEOUT


@dataclass(frozen=True)
class SarabanSettings:
    types: tuple[str, ...] = tuple(SARABAN_TYPES)
    render_dpi: int = SARABAN_RENDER_DPI
    target_per_type: int = SARABAN_TARGET_PER_TYPE
    max_pages_per_pdf: int = SARABAN_MAX_PAGES_PER_PDF
    scanned_ratio: float = SARABAN_SCANNED_RATIO
    pii_title_blocklist: tuple[str, ...] = tuple(PII_TITLE_BLOCKLIST)


@dataclass(frozen=True)
class GoogleCrawlSettings:
    crawl_dir: Path = GOOGLE_CRAWL_DIR
    blocked_domains: tuple[str, ...] = GOOGLE_BLOCKED_DOMAINS
    default_limit: int = 20
    default_min_dim: int = 800


@dataclass(frozen=True)
class Credentials:
    datagoth_api_key: str
    hf_token: str


def _parse_env_file(env_path: Path | str = ".env") -> tuple[str, str]:
    """Parse data.go.th API key and Hugging Face token from a .env file."""
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
            if k_clean in {"thai-ocr", "hf_token", "huggingface_token", "thai_ocr_hf_token"}:
                hf_token = v_clean
            elif k_clean in {"datagoth_api_key", "data_go_th_key", "api_key"}:
                datagoth_key = v_clean
        elif not datagoth_key:
            datagoth_key = line

    return datagoth_key, hf_token


def load_datagoth_api_key(env_path: Path | str = ".env") -> str:
    """Load data.go.th API key from environment variables or .env without requiring HF_TOKEN."""
    for var in ("DATAGOTH_API_KEY", "DATA_GO_TH_KEY"):
        val = os.environ.get(var, "").strip()
        if val:
            return val

    datagoth_key, _ = _parse_env_file(env_path)
    if not datagoth_key:
        raise ValueError("data.go.th API key not found in environment or .env")
    return datagoth_key


def load_hf_token(env_path: Path | str = ".env") -> str:
    """Load Hugging Face token from environment variables or .env without requiring DATAGOTH_API_KEY."""
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "THAI_OCR_HF_TOKEN"):
        val = os.environ.get(var, "").strip()
        if val:
            return val

    _, hf_token = _parse_env_file(env_path)
    if not hf_token:
        raise ValueError("Hugging Face token (thai-ocr / HF_TOKEN) not found in environment or .env")
    return hf_token


def load_credentials(env_path: Path | str = ".env") -> Credentials:
    """Load both data.go.th API key and Hugging Face token (backward-compatible helper)."""
    return Credentials(
        datagoth_api_key=load_datagoth_api_key(env_path),
        hf_token=load_hf_token(env_path),
    )


def load_gt_api_keys(env_path: Path | str = ".env") -> tuple[str, str]:
    """Load (gemini_api_key, iapp_api_key) from environment variables or .env."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    iapp_key = (
        os.environ.get("IAPP_API_KEY", "").strip()
        or os.environ.get("OPENTHAI_API_KEY", "").strip()
        or os.environ.get("GT_THAI_DOCS", "").strip()
    )

    path = Path(env_path)
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k_clean = k.strip().upper()
            v_clean = v.strip().strip('"').strip("'")
            if k_clean in {"GEMINI_API_KEY", "GOOGLE_API_KEY"} and not gemini_key:
                gemini_key = v_clean
            elif k_clean in {"IAPP_API_KEY", "OPENTHAI_API_KEY", "GT_THAI_DOCS"} and not iapp_key:
                iapp_key = v_clean

    return gemini_key, iapp_key


def load_kku_api_key(env_path: Path | str = ".env") -> str:
    """Load KKU GenAI API key (KKU_API / KKU_API_KEY / OPENAI_API_KEY)."""
    for var in ("KKU_API", "KKU_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val:
            return val

    path = Path(env_path)
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip().upper() in {"KKU_API", "KKU_API_KEY", "OPENAI_API_KEY"}:
                return v.strip().strip('"').strip("'")

    raise ValueError("KKU_API key not found in environment or .env")
