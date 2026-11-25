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
    INTERN_SODA_DIR,
    INTERN_SODA_KRUT_HF_DIR,
    RAJAKITCHA_DIR,
    RAJAKITCHA_HF_DIR,
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
    krut_path = "data/intern_soda_krut/train-*" if any(f.startswith("data/intern_soda_krut/") for f in repo_files) else "intern_soda_krut/train-*"
    if "intern_soda_krut" not in existing_configs and any(
        f.startswith(krut_path.split("*")[0]) for f in repo_files
    ):
        existing_configs["intern_soda_krut"] = {
            "config_name": "intern_soda_krut",
            "data_files": [{"split": "train", "path": krut_path}],
        }
    rajakitcha_path = "data/rajakitcha/train-*" if any(f.startswith("data/rajakitcha/") for f in repo_files) else "rajakitcha/train-*"
    if "rajakitcha" not in existing_configs and any(
        f.startswith(rajakitcha_path.split("*")[0]) for f in repo_files
    ):
        existing_configs["rajakitcha"] = {
            "config_name": "rajakitcha",
            "data_files": [{"split": "train", "path": rajakitcha_path}],
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
    intern_soda_krut_stats: dict | None = None,
    rajakitcha_stats: dict | None = None,
) -> str:
    """Build a Dataset Card covering all benchmark & training configs."""
    yaml_str = merge_hub_yaml(hub_yaml, repo_files or [])

    sections: list[str] = [
        f"---\n{yaml_str}\n---\n",
        "# Thai Government PDF OCR Test Benchmark (`narawit16/thai-govern-test-ocr`)\n",
        "An evaluation benchmark and training corpus of **Thai government documents** designed for OCR, layout analysis, and VLM document-parsing:\n",
        "- **`default`**: Broad cross-agency Thai open government PDFs from `data.go.th` (15 domain groups, 111 agencies, 200 DPI, **111 `human_verified`** GT).\n",
        "- **`saraban_6types`**: Curated **6 official Thai Saraban correspondence types** (*ระเบียบสำนักนายกรัฐมนตรีว่าด้วยงานสารบรรณ*) from `dla.go.th` & `data.go.th` (300 DPI, PDPA-screened, with human-verified Golden Set Ground Truth).\n",
        "- **`google_docs`**: Web-crawled Thai government & local administration document images (108 images, **96 `human_verified`** GT).\n",
        "- **`dla_bulk`**: High-coverage multi-portal collection from `www.dla.go.th` & `sll.dla.go.th` (official letters, local laws, articles, legal opinions, 200 DPI, PDPA-screened).\n",
        "- **`intern_soda_krut`**: Thai government documents containing the official Garuda emblem (*ตราครุฑ*), contributed by **Intern-SODA** (`train_kruthdetect/DATA/has_krut`, 446 images, **120 `human_verified`** GT).\n",
        "- **`rajakitcha`**: Royal Gazette (*ราชกิจจานุเบกษา*) constituency announcement pages from `KittapatR/Thai-OCR-open-source-initiative` (87 images, **20 `human_verified`** GT).\n",
    ]

    # Top-level Human-Verified Ground Truth Summary
    saraban_hv, saraban_fl = 0, 0
    saraban_ann_path = SARABAN_DIR / "golden_120" / "gt_annotations.jsonl"
    if saraban_ann_path.exists():
        s_anns = [json.loads(l) for l in saraban_ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        saraban_hv = sum(1 for a in s_anns if a.get("gt_status") == "human_verified" and (a.get("ground_truth_text") or "").strip())
        saraban_fl = sum(1 for a in s_anns if a.get("gt_status") == "flagged")

    google_hv, google_fl = 0, 0
    google_ann_path = GOOGLE_CRAWL_DIR / "gt" / "gt_annotations.jsonl"
    if google_ann_path.exists():
        g_anns = [json.loads(l) for l in google_ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        google_hv = sum(1 for a in g_anns if a.get("gt_status") == "human_verified" and (a.get("ground_truth_text") or "").strip())
        google_fl = sum(1 for a in g_anns if a.get("gt_status") == "flagged")

    default_hv, default_fl = 0, 0
    default_ann_path = DATA_DIR / "default_gt" / "gt_annotations.jsonl"
    if default_ann_path.exists():
        d_anns = [json.loads(l) for l in default_ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        default_hv = sum(1 for a in d_anns if a.get("gt_status") == "human_verified" and (a.get("ground_truth_text") or "").strip())
        default_fl = sum(1 for a in d_anns if a.get("gt_status") == "flagged")

    rajakitcha_hv, rajakitcha_fl = 0, 0
    rajakitcha_ann_path = RAJAKITCHA_DIR / "gt" / "gt_annotations.jsonl"
    if rajakitcha_ann_path.exists():
        r_anns = [json.loads(l) for l in rajakitcha_ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rajakitcha_hv = sum(1 for a in r_anns if a.get("gt_status") == "human_verified" and (a.get("ground_truth_text") or "").strip())
        rajakitcha_fl = sum(1 for a in r_anns if a.get("gt_status") == "flagged")

    krut_hv, krut_fl = 0, 0
    krut_ann_path = INTERN_SODA_DIR / "gt_krut" / "gt_annotations.jsonl"
    if krut_ann_path.exists():
        k_anns = [json.loads(l) for l in krut_ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        krut_hv = sum(1 for a in k_anns if a.get("gt_status") == "human_verified" and (a.get("ground_truth_text") or "").strip())
        krut_fl = sum(1 for a in k_anns if a.get("gt_status") == "flagged")

    dla_hv, dla_fl = 0, 0
    dla_ann_path = DLA_BULK_DIR / "gt" / "gt_annotations.jsonl"
    if dla_ann_path.exists():
        dla_anns = [json.loads(l) for l in dla_ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        dla_hv = sum(1 for a in dla_anns if a.get("gt_status") == "human_verified" and (a.get("ground_truth_text") or "").strip())
        dla_fl = sum(1 for a in dla_anns if a.get("gt_status") == "flagged")

    total_hv = saraban_hv + google_hv + default_hv + rajakitcha_hv + krut_hv + dla_hv
    if total_hv > 0:
        s_total = saraban_stats["total_pages"] if saraban_stats else 419
        g_total = google_stats["total_pages"] if google_stats else 108
        d_total = default_stats["total_pages"] if default_stats else 319
        r_total = rajakitcha_stats["total_pages"] if rajakitcha_stats else 87
        k_total = intern_soda_krut_stats["total_pages"] if intern_soda_krut_stats else 446
        dla_total = dla_bulk_stats["total_pages"] if dla_bulk_stats else 1649
        sections.append(
            f"""## Human-Verified Ground Truth Summary

Total **`human_verified`** pages/images across all configs: **{total_hv:,}**

| Config | Split | Total | `human_verified` | `flagged` | `unverified` | Draft / Verification Workflow |
|---|---|---:|---:|---:|---:|---|
| `saraban_6types` | `test` | {s_total:,} | **{saraban_hv:,}** | {saraban_fl:,} | {max(0, s_total - saraban_hv - saraban_fl):,} | Golden Set (`is_golden_set == True`) dual-drafted + human-verified via Review UI |
| `default` | `test` | {d_total:,} | **{default_hv:,}** | {default_fl:,} | {max(0, d_total - default_hv - default_fl):,} | Stratified 120-page subset dual-drafted via KKU VLMs + human-verified via Review UI |
| `google_docs` | `test` | {g_total:,} | **{google_hv:,}** | {google_fl:,} | {max(0, g_total - google_hv - google_fl):,} | Drafted with **iApp OpenThai OCR 2.1** + human-verified via Review UI |
| `dla_bulk` | `train` | {dla_total:,} | **{dla_hv:,}** | {dla_fl:,} | {max(0, dla_total - dla_hv - dla_fl):,} | Stratified 120-page subset drafted with KKU VLMs & **iApp OpenThai OCR 2.1** + human-verified |
| `intern_soda_krut` | `train` | {k_total:,} | **{krut_hv:,}** | {krut_fl:,} | {max(0, k_total - krut_hv - krut_fl):,} | Garuda-stamped 120-image subset dual-drafted via KKU VLMs + human-verified (credit: Intern-SODA) |
| `rajakitcha` | `train` | {r_total:,} | **{rajakitcha_hv:,}** | {rajakitcha_fl:,} | {max(0, r_total - rajakitcha_hv - rajakitcha_fl):,} | Royal Gazette (*ราชกิจจานุเบกษา*) drafted with **iApp OpenThai OCR 2.1** + human-verified via Review UI |

> **Tip:** Filter verified Ground Truth rows directly with `ds.filter(lambda x: x["gt_status"] == "human_verified")`. Exclude `gt_status == "flagged"` (illegible/corrupted/unsuitable pages) during OCR accuracy evaluation.
"""
        )

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
        gt_summary_block = ""
        gt_ann_path = Path("data/saraban/golden_120/gt_annotations.jsonl")
        eval_md_path = Path("data/saraban/golden_120/eval_report.md")
        if gt_ann_path.exists():
            anns = [json.loads(l) for l in gt_ann_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            hv = sum(1 for a in anns if a.get("gt_status") == "human_verified" and (a.get("ground_truth_text") or "").strip())
            fl = sum(1 for a in anns if a.get("gt_status") == "flagged")
            gt_summary_block = f"""
### Golden Set Ground Truth (`is_golden_set == True`)
- **Verified GT Pages (`gt_status == "human_verified"`):** {hv:,} pages (`flagged`: {fl:,})
- **GT Format:** Full-page Markdown (GFM tables + HTML `<table>` for merged cells) in `ground_truth_text`, plus KIE Header JSON (`doc_no`, `doc_date`, `subject`) in `gt_kie_json`.
- **Transcription Rules:**
  - Exact printed numerals preserved verbatim (`๑–๙` vs `0–9`).
  - Printed body/headers + clearly readable rubber stamps (`ด่วนที่สุด`, agency receipt stamps) transcribed; illegible wet signatures tagged `[SIGNATURE]`; handwritten margin notes excluded.
"""
        if eval_md_path.exists():
            eval_lines = [
                ln for ln in eval_md_path.read_text(encoding="utf-8").splitlines()
                if not ln.startswith("# ")
            ]
            gt_summary_block += (
                "\n### Baseline Draft Evaluation (on `human_verified` Golden Set)\n"
                "*(Note: `gemini-4-hydrogen` served as the primary UI pre-fill draft, so its baseline CER reflects draft-assisted edits.)*\n\n"
                + "\n".join(eval_lines[:9])
                + "\n"
            )

        sections.append(
            f"""## Config 1: `saraban_6types` (6 Official Saraban Document Types)

- **Total Pages (`test` split):** {saraban_stats['total_pages']:,} pages from {saraban_stats['total_documents']:,} PDFs ({saraban_stats['total_organizations']:,} organizations)
- **Resolution:** {saraban_stats['dpi']} DPI (RGB PNG)
- **Stratification:** {saraban_stats['scanned_share_pct']}% Scanned (`scanned` + `scanned_ocr`) / {saraban_stats['digital_share_pct']}% Complex Digital
- **PDPA Screening:** Pre-filtered HR/welfare titles + 13-digit Arabic & Thai numeral ID checksum check on both embedded text and RapidOCR scanned text (`ocr_pii_rejected={saraban_stats.get('ocr_pii_rejected', 0)}`).
{gt_summary_block}

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
        default_gt_block = ""
        if default_ann_path.exists():
            default_gt_block = f"""
### Human-Verified Ground Truth
- **Verified GT Pages (`gt_status == "human_verified"`):** {default_hv:,} / {default_stats['total_pages']:,} pages (`flagged`: {default_fl:,})
- **Draft Engine & Verification:** Stratified 120-page subset dual-drafted via KKU VLMs (`gemini-4-hydrogen`, `gemma4-26b`, `qwen3.8-27b`), then human-verified via Review UI.
- **GT Columns:** Full-page Markdown in `ground_truth_text`, KIE Header JSON (`doc_no`, `doc_date`, `subject`) in `gt_kie_json`, and `quality_tags`.
"""
        sections.append(
            f"""## Config 2: `default` (General `data.go.th` Multi-Domain Benchmark)

- **Total Pages (`test` split):** {default_stats['total_pages']:,} pages from {default_stats['total_documents']:,} PDFs ({default_stats['total_organizations']:,} agencies)
- **Resolution:** {default_stats['dpi']} DPI (RGB PNG)
- **Stratification:** {default_stats['scanned_share_pct']}% Scanned / {default_stats['digital_share_pct']}% Complex Digital
- **Document-Only Filtering:** Excluded {default_stats.get('graphic_rejected_pages', 0):,} non-document graphic pages (posters, infographics, colorful covers, 16:9 slides).
{default_gt_block}
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
        google_gt_block = ""
        if google_ann_path.exists():
            google_gt_block = f"""
### Human-Verified Ground Truth
- **Verified GT Images (`gt_status == "human_verified"`):** {google_hv:,} / {google_stats['total_pages']:,} images (`flagged`: {google_fl:,})
- **Draft Engine & Verification:** Pre-filled with **iApp OpenThai OCR 2.1** (`api.iapp.co.th/v3/store/ocr/document/markdown`), then reviewed and corrected page-by-page via the Human-in-the-Loop Review UI.
- **GT Columns:** Full-page Markdown in `ground_truth_text`, KIE Header JSON (`doc_no`, `doc_date`, `subject`) in `gt_kie_json`, and document quality tags in `quality_tags`.
"""
        sections.append(
            f"""## Config 3: `google_docs` (Web-Crawled Thai Government & Local Admin Documents)

- **Total Images (`test` split):** {google_stats['total_pages']:,} images (EXIF-normalized RGB PNG, deduplicated, PDPA-screened)
- **Max Edge:** {google_stats.get('max_edge', 3508)} px
{google_gt_block}
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
        dla_gt_block = ""
        if dla_ann_path.exists():
            dla_gt_block = f"""
### Human-Verified Ground Truth
- **Verified GT Pages (`gt_status == "human_verified"`):** {dla_hv:,} / {dla_bulk_stats['total_pages']:,} pages (`flagged`: {dla_fl:,})
- **Draft Engine & Verification:** Stratified 120-page subset drafted via KKU VLMs and **iApp OpenThai OCR 2.1**, then human-verified via Review UI.
- **GT Columns:** Full-page Markdown in `ground_truth_text`, KIE Header JSON (`doc_no`, `doc_date`, `subject`) in `gt_kie_json`, and `quality_tags`.
"""
        sections.append(
            f"""## Config 4: `dla_bulk` (DLA Multi-Portal Corpus from `dla.go.th` & `sll.dla.go.th`)

- **Total Pages (`{dla_bulk_stats.get('split', 'train')}` split):** {dla_bulk_stats['total_pages']:,} pages from {dla_bulk_stats['total_documents']:,} PDFs
- **Resolution:** {dla_bulk_stats['dpi']} DPI (RGB PNG)
- **Stratification:** {dla_bulk_stats['scanned_share_pct']}% Scanned (`scanned` + `scanned_ocr`) / {dla_bulk_stats['digital_share_pct']}% Digital
- **PDPA & Graphic Screening:** Excluded {dla_bulk_stats.get('pii_excluded_pages', 0):,} PII-flagged pages, {dla_bulk_stats.get('other_excluded_pages', 0):,} blank/other pages, and {dla_bulk_stats.get('graphic_excluded_pages', 0):,} non-document graphic pages (posters, infographics, colorful covers, 16:9 slides).
{dla_gt_block}
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

    if intern_soda_krut_stats:
        krut_gt_block = ""
        if krut_ann_path.exists():
            krut_gt_block = f"""
### Human-Verified Ground Truth
- **Verified GT Images (`gt_status == "human_verified"`):** {krut_hv:,} / {intern_soda_krut_stats['total_pages']:,} images (`flagged`: {krut_fl:,})
- **Draft Engine & Verification:** 120-image diverse subset dual-drafted via KKU VLMs (`gemini-4-hydrogen` + `gemma4-26b`), then human-verified via Review UI.
- **GT Columns:** Full-page Markdown in `ground_truth_text`, KIE Header JSON (`doc_no`, `doc_date`, `subject`) in `gt_kie_json`, and `quality_tags`.
"""
        sections.append(
            f"""## Config 5: `intern_soda_krut` (Garuda-Stamped Thai Official Documents)

- **Total Images (`{intern_soda_krut_stats.get('split', 'train')}` split):** {intern_soda_krut_stats['total_pages']:,} images (EXIF-normalized RGB PNG, deduplicated, `has_krut=True`)
- **Max Edge:** {intern_soda_krut_stats.get('max_edge', 3508)} px
- **Source & Credit:** **Intern-SODA** (`train_kruthdetect/DATA/has_krut`) — Thai government documents featuring the official Garuda emblem (*ตราครุฑ*).
{krut_gt_block}"""
        )

    if rajakitcha_stats:
        sections.append(
            f"""## Config 6: `rajakitcha` (Royal Gazette / ราชกิจจานุเบกษา)

- **Total Images (`{rajakitcha_stats.get('split', 'train')}` split):** {rajakitcha_stats['total_pages']:,} images (EXIF-normalized RGB PNG, deduplicated)
- **Max Edge:** {rajakitcha_stats.get('max_edge', 3508)} px
- **Source:** `KittapatR/Thai-OCR-open-source-initiative` (`Example/Constituencies`) — Royal Gazette (*ราชกิจจานุเบกษา*) constituency announcement pages.
"""
        )

    sections.append(
        """## Acknowledgments & Credits

- **Intern-SODA**: Contributed the `intern_soda_krut` subset (Garuda-stamped Thai official document pages from `train_kruthdetect/DATA/has_krut`).

## Usage

```python
from datasets import load_dataset

# Load the 6-type Thai Saraban correspondence benchmark (300 DPI)
saraban = load_dataset("narawit16/thai-govern-test-ocr", "saraban_6types", split="test")
scanned_letters = saraban.filter(
    lambda x: x["saraban_type"] == "external_letter" and x["page_type"] == "scanned"
)

# Load only human-verified Ground Truth samples from saraban_6types or google_docs
saraban_gt = saraban.filter(lambda x: x["gt_status"] == "human_verified")
google_gt = load_dataset("narawit16/thai-govern-test-ocr", "google_docs", split="test").filter(
    lambda x: x["gt_status"] == "human_verified"
)

# Load the general data.go.th benchmark (200 DPI) and its human-verified GT subset
general = load_dataset("narawit16/thai-govern-test-ocr", "default", split="test")
general_gt = general.filter(lambda x: x["gt_status"] == "human_verified")
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
        elif config_name == "intern_soda_krut":
            ds_dir = INTERN_SODA_KRUT_HF_DIR
            stats_path = INTERN_SODA_DIR / "dataset_stats_krut.json"
        elif config_name == "rajakitcha":
            ds_dir = RAJAKITCHA_HF_DIR
            stats_path = RAJAKITCHA_DIR / "dataset_stats.json"
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
    intern_soda_krut_stats_path = INTERN_SODA_DIR / "dataset_stats_krut.json"
    rajakitcha_stats_path = RAJAKITCHA_DIR / "dataset_stats.json"
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
    intern_soda_krut_stats = (
        json.loads(intern_soda_krut_stats_path.read_text(encoding="utf-8"))
        if intern_soda_krut_stats_path.exists()
        else None
    )
    rajakitcha_stats = (
        json.loads(rajakitcha_stats_path.read_text(encoding="utf-8"))
        if rajakitcha_stats_path.exists()
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
            intern_soda_krut_stats=intern_soda_krut_stats,
            rajakitcha_stats=rajakitcha_stats,
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
        choices=["default", "saraban_6types", "google_docs", "dla_bulk", "intern_soda_krut", "rajakitcha"],
        default="default",
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Only regenerate and upload README.md without re-pushing dataset shards",
    )
    args = parser.parse_args(argv)
    push_dataset(config_name=args.config_name, readme_only=args.readme_only)
