"""Push prepared Thai Government OCR Benchmark datasets and Dataset Card to Hugging Face."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from train_ocr.common.settings import (
    DATA_DIR,
    DLA_BULK_DIR,
    DLA_BULK_HF_DIR,
    GOOGLE_CRAWL_DIR,
    GOOGLE_HF_DIR,
    HF_DATASET_DIR,
    HF_REPO_ID,
    SARABAN_DIR,
    SARABAN_HF_DIR,
    load_hf_token,
)


def extract_yaml_frontmatter(readme_text: str) -> str:
    """Extract YAML frontmatter block from an existing Hub README.md."""
    m = re.match(r"^---\s*\n(.*?\n)---\s*\n", readme_text, re.S)
    return m.group(1) if m else ""


def merge_hub_yaml(hub_yaml: str, repo_files: list[str]) -> str:
    """Ensure YAML frontmatter explicitly declares both default and saraban_6types configs."""
    meta = yaml.safe_load(hub_yaml) if hub_yaml.strip() else {}
    if not isinstance(meta, dict):
        meta = {}

    meta.setdefault("language", ["th", "en"])
    meta.setdefault("license", ["odc-by", "cc-by-4.0"])
    meta.setdefault("task_categories", ["image-to-text"])
    meta.setdefault(
        "tags",
        ["ocr", "thai", "government-documents", "pdf", "benchmark", "data-go-th", "saraban"],
    )
    if "saraban" not in meta["tags"]:
        meta["tags"].append("saraban")
    meta["pretty_name"] = "Thai Government PDF OCR Test Benchmark"

    default_path = "data/test-*"
    saraban_path = "saraban_6types/test-*"
    google_path = "google_docs/test-*"
    if any(f.startswith("default/test-") for f in repo_files):
        default_path = "default/test-*"
    if any(f.startswith("data/saraban_6types/") for f in repo_files):
        saraban_path = "data/saraban_6types/test-*"
    if any(f.startswith("data/google_docs/") for f in repo_files):
        google_path = "data/google_docs/test-*"

    existing_configs = {
        c["config_name"]: c
        for c in meta.get("configs", [])
        if isinstance(c, dict) and "config_name" in c
    }
    if "default" not in existing_configs and any(
        f.startswith(default_path.split("*")[0]) for f in repo_files
    ):
        existing_configs["default"] = {
            "config_name": "default",
            "data_files": [{"split": "test", "path": default_path}],
        }
    if "saraban_6types" not in existing_configs and any(
        f.startswith(saraban_path.split("*")[0]) for f in repo_files
    ):
        existing_configs["saraban_6types"] = {
            "config_name": "saraban_6types",
            "data_files": [{"split": "test", "path": saraban_path}],
        }
    if "google_docs" not in existing_configs and any(
        f.startswith(google_path.split("*")[0]) for f in repo_files
    ):
        existing_configs["google_docs"] = {
            "config_name": "google_docs",
            "data_files": [{"split": "test", "path": google_path}],
        }
    dla_bulk_split = "train" if any(f.startswith("dla_bulk/train-") or f.startswith("data/dla_bulk/train-") for f in repo_files) else "test"
    dla_bulk_path = f"data/dla_bulk/{dla_bulk_split}-*" if any(f.startswith("data/dla_bulk/") for f in repo_files) else f"dla_bulk/{dla_bulk_split}-*"
    if "dla_bulk" not in existing_configs and any(
        f.startswith(dla_bulk_path.split("*")[0]) for f in repo_files
    ):
        existing_configs["dla_bulk"] = {
            "config_name": "dla_bulk",
            "data_files": [{"split": dla_bulk_split, "path": dla_bulk_path}],
        }
    if existing_configs:
        meta["configs"] = list(existing_configs.values())

    return yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()


def build_combined_readme(
    default_stats: dict | None,
    saraban_stats: dict | None,
    hub_yaml: str = "",
    repo_files: list[str] | None = None,
    google_stats: dict | None = None,
    dla_bulk_stats: dict | None = None,
) -> str:
    """Build a Dataset Card covering all benchmark & training configs."""
    yaml_str = merge_hub_yaml(hub_yaml, repo_files or [])

    sections: list[str] = [
        f"---\n{yaml_str}\n---\n",
        "# Thai Government PDF OCR Test Benchmark (`narawit16/thai-govern-test-ocr`)\n",
        "An evaluation benchmark and training corpus of **Thai government documents** designed for OCR, layout analysis, and VLM document-parsing:\n",
        "- **`default`**: Broad cross-agency Thai open government PDFs from `data.go.th` (15 domain groups, 111 agencies, 200 DPI).\n",
        "- **`saraban_6types`**: Curated **6 official Thai Saraban correspondence types** (*ระเบียบสำนักนายกรัฐมนตรีว่าด้วยงานสารบรรณ*) from `dla.go.th` & `data.go.th` (300 DPI, PDPA-screened, annotation-ready).\n",
        "- **`google_docs`**: Web-crawled Thai government & local administration document images.\n",
        "- **`dla_bulk`**: High-coverage multi-portal collection from `www.dla.go.th` & `sll.dla.go.th` (official letters, local laws, articles, legal opinions, 200 DPI, PDPA-screened).\n",
    ]

    if saraban_stats:
        st_rows = "\n".join(
            f"| `{st}` | {cnt:,} |"
            for st, cnt in sorted(
                saraban_stats["by_saraban_type"].items(), key=lambda x: x[1], reverse=True
            )
        )
        pt_rows = "\n".join(
            f"| `{pt}` | {cnt:,} ({100.0 * cnt / saraban_stats['total_pages']:.1f}%) |"
            for pt, cnt in sorted(
                saraban_stats["by_page_type"].items(), key=lambda x: x[1], reverse=True
            )
        )
        role_rows = "\n".join(
            f"| `{role}` | {cnt:,} |"
            for role, cnt in sorted(
                saraban_stats["by_page_role"].items(), key=lambda x: x[1], reverse=True
            )
        )
        sections.append(
            f"""## Config 1: `saraban_6types` (6 Official Saraban Document Types)

- **Total Pages (`test` split):** {saraban_stats['total_pages']:,} pages from {saraban_stats['total_documents']:,} PDFs ({saraban_stats['total_organizations']:,} organizations)
- **Resolution:** {saraban_stats['dpi']} DPI (RGB PNG)
- **Stratification:** {saraban_stats['scanned_share_pct']}% Scanned (`scanned` + `scanned_ocr`) / {saraban_stats['digital_share_pct']}% Complex Digital
- **PDPA Screening:** Pre-filtered HR/welfare titles + 13-digit Arabic & Thai numeral ID checksum check on both embedded text and RapidOCR scanned text (`ocr_pii_rejected={saraban_stats.get('ocr_pii_rejected', 0)}`).

### Breakdown by `saraban_type`
| `saraban_type` | Pages |
|---|---:|
{st_rows}

### Breakdown by `page_type` & `page_role`
| `page_type` | Pages |
|---|---:|
{pt_rows}

| `page_role` | Pages |
|---|---:|
{role_rows}
"""
        )

    if default_stats:
        group_rows = "\n".join(
            f"| `{grp}` | {cnt:,} |"
            for grp, cnt in sorted(
                default_stats["by_group"].items(), key=lambda x: x[1], reverse=True
            )
        )
        pt_rows = "\n".join(
            f"| `{cat}` | {cnt:,} ({100.0 * cnt / default_stats['total_pages']:.1f}%) |"
            for cat, cnt in sorted(
                default_stats["by_page_type"].items(), key=lambda x: x[1], reverse=True
            )
        )
        sections.append(
            f"""## Config 2: `default` (General `data.go.th` Multi-Domain Benchmark)

- **Total Pages (`test` split):** {default_stats['total_pages']:,} pages from {default_stats['total_documents']:,} PDFs ({default_stats['total_organizations']:,} agencies)
- **Resolution:** {default_stats['dpi']} DPI (RGB PNG)
- **Stratification:** {default_stats['scanned_share_pct']}% Scanned / {default_stats['digital_share_pct']}% Complex Digital

### Page Type Breakdown
| `page_type` | Pages |
|---|---:|
{pt_rows}

### Domain Group Breakdown (`group`)
| Group (`data.go.th`) | Pages |
|---|---:|
{group_rows}
"""
        )

    if google_stats:
        kw_rows = "\n".join(
            f"| `{kw}` | {cnt:,} |"
            for kw, cnt in sorted(
                google_stats["by_keyword"].items(), key=lambda x: x[1], reverse=True
            )
        )
        sections.append(
            f"""## Config 3: `google_docs` (Web-Crawled Thai Government & Local Admin Documents)

- **Total Images (`test` split):** {google_stats['total_pages']:,} images (EXIF-normalized RGB PNG, deduplicated, PDPA-screened)
- **Max Edge:** {google_stats.get('max_edge', 3508)} px

### Breakdown by Search Keyword
| `keyword` | Images |
|---|---:|
{kw_rows}
"""
        )

    if dla_bulk_stats:
        src_rows = "\n".join(
            f"| `{src}` | {cnt:,} |"
            for src, cnt in sorted(
                dla_bulk_stats["by_source"].items(), key=lambda x: x[1], reverse=True
            )
        )
        pt_rows = "\n".join(
            f"| `{pt}` | {cnt:,} ({100.0 * cnt / max(dla_bulk_stats['total_pages'], 1):.1f}%) |"
            for pt, cnt in sorted(
                dla_bulk_stats["by_page_type"].items(), key=lambda x: x[1], reverse=True
            )
        )
        sections.append(
            f"""## Config 4: `dla_bulk` (DLA Multi-Portal Corpus from `dla.go.th` & `sll.dla.go.th`)

- **Total Pages (`{dla_bulk_stats.get('split', 'train')}` split):** {dla_bulk_stats['total_pages']:,} pages from {dla_bulk_stats['total_documents']:,} PDFs
- **Resolution:** {dla_bulk_stats['dpi']} DPI (RGB PNG)
- **Stratification:** {dla_bulk_stats['scanned_share_pct']}% Scanned (`scanned` + `scanned_ocr`) / {dla_bulk_stats['digital_share_pct']}% Digital
- **PDPA Screening:** Excluded {dla_bulk_stats.get('pii_excluded_pages', 0):,} PII-flagged pages and {dla_bulk_stats.get('other_excluded_pages', 0):,} blank/other pages.

### Breakdown by Portal (`source`)
| `source` | Pages |
|---|---:|
{src_rows}

### Breakdown by `page_type`
| `page_type` | Pages |
|---|---:|
{pt_rows}
"""
        )

    sections.append(
        """## Usage

```python
from datasets import load_dataset

# Load the 6-type Thai Saraban correspondence benchmark (300 DPI)
saraban = load_dataset("narawit16/thai-govern-test-ocr", "saraban_6types", split="test")
scanned_letters = saraban.filter(
    lambda x: x["saraban_type"] == "external_letter" and x["page_type"] == "scanned"
)

# Load the general data.go.th benchmark (200 DPI)
general = load_dataset("narawit16/thai-govern-test-ocr", "default", split="test")
```

> **Note on KIE Evaluation (`meta_doc_no`, `meta_doc_date`, `meta_subject`):**
> Web metadata often uses Arabic digits (e.g., `มท 0808.2/ว 1234`), whereas printed/stamped paper documents frequently use Thai numerals (`มท ๐๘๐๘.๒/ว ๑๒๓๔`). Normalize Thai digits `๐–๙` → `0–9` before comparing extracted header fields against `meta_*` columns.
"""
    )
    return "\n".join(sections)


def push_dataset(config_name: str = "default", readme_only: bool = False) -> None:
    """Push a prepared dataset config to Hugging Face Hub using HF_TOKEN only."""
    from datasets import DatasetDict
    from huggingface_hub import HfApi, hf_hub_download

    hf_token = load_hf_token()
    if not readme_only:
        if config_name == "saraban_6types":
            ds_dir = SARABAN_HF_DIR
            stats_path = SARABAN_DIR / "dataset_stats.json"
        elif config_name == "google_docs":
            ds_dir = GOOGLE_HF_DIR
            stats_path = GOOGLE_CRAWL_DIR / "dataset_stats.json"
        elif config_name == "dla_bulk":
            ds_dir = DLA_BULK_HF_DIR
            stats_path = DLA_BULK_DIR / "dataset_stats.json"
        else:
            ds_dir = HF_DATASET_DIR
            stats_path = DATA_DIR / "dataset_stats.json"

        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        ds_dict = DatasetDict.load_from_disk(str(ds_dir))

        print(f"Pushing config='{config_name}' ({stats['total_pages']} pages) to {HF_REPO_ID}...")
        ds_dict.push_to_hub(
            HF_REPO_ID,
            config_name=config_name,
            token=hf_token,
            commit_message=(
                f"Upload config={config_name} "
                f"({stats['total_pages']} pages from {stats['total_documents']} PDFs)"
            ),
        )

    hub_yaml = ""
    try:
        dl_readme = hf_hub_download(
            repo_id=HF_REPO_ID, filename="README.md", repo_type="dataset", token=hf_token
        )
        hub_yaml = extract_yaml_frontmatter(Path(dl_readme).read_text(encoding="utf-8"))
    except Exception:
        pass

    default_stats_path = DATA_DIR / "dataset_stats.json"
    saraban_stats_path = SARABAN_DIR / "dataset_stats.json"
    google_stats_path = GOOGLE_CRAWL_DIR / "dataset_stats.json"
    dla_bulk_stats_path = DLA_BULK_DIR / "dataset_stats.json"
    default_stats = (
        json.loads(default_stats_path.read_text(encoding="utf-8"))
        if default_stats_path.exists()
        else None
    )
    saraban_stats = (
        json.loads(saraban_stats_path.read_text(encoding="utf-8"))
        if saraban_stats_path.exists()
        else None
    )
    google_stats = (
        json.loads(google_stats_path.read_text(encoding="utf-8"))
        if google_stats_path.exists()
        else None
    )
    dla_bulk_stats = (
        json.loads(dla_bulk_stats_path.read_text(encoding="utf-8"))
        if dla_bulk_stats_path.exists()
        else None
    )

    api = HfApi(token=hf_token)
    repo_files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")

    readme_path = DATA_DIR / "README.md"
    readme_path.write_text(
        build_combined_readme(
            default_stats,
            saraban_stats,
            hub_yaml=hub_yaml,
            repo_files=repo_files,
            google_stats=google_stats,
            dla_bulk_stats=dla_bulk_stats,
        ),
        encoding="utf-8",
    )

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"Update Dataset Card README.md for config={config_name}",
    )
    print(f"Successfully updated README.md on https://huggingface.co/datasets/{HF_REPO_ID}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Push dataset config to Hugging Face Hub.")
    parser.add_argument(
        "--config-name",
        choices=["default", "saraban_6types", "google_docs", "dla_bulk"],
        default="default",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Only regenerate and upload README.md without re-pushing dataset shards",
    )
    args = parser.parse_args(argv)
    push_dataset(config_name=args.config_name, readme_only=args.readme_only)
