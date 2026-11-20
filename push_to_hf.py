"""Push prepared Thai Government OCR Benchmark dataset and Dataset Card to Hugging Face."""

import json
from pathlib import Path

from datasets import DatasetDict
from huggingface_hub import HfApi

from config import DATA_DIR, HF_DATASET_DIR, HF_REPO_ID, load_credentials


def build_readme(stats: dict) -> str:
    """Build a Hugging Face Dataset Card (README.md) from dataset_stats.json."""
    group_rows = "\n".join(
        f"| `{grp}` | {cnt:,} |"
        for grp, cnt in sorted(stats["by_group"].items(), key=lambda x: x[1], reverse=True)
    )
    type_descs = {
        "scanned": "Pure scanned image page (`<20` extracted chars, `>50%` image coverage)",
        "scanned_ocr": "Scanned page with pre-existing OCR text layer (`>=20` chars, `>50%` image coverage)",
        "digital": "Complex native digital page (`>=50` chars, `<=50%` image coverage, high table/layout score)",
    }
    cat_rows = "\n".join(
        f"| `{cat}` | {type_descs.get(cat, '')} | {cnt:,} ({100.0 * cnt / stats['total_pages']:.1f}%) |"
        for cat, cnt in sorted(stats["by_page_type"].items(), key=lambda x: x[1], reverse=True)
    )
    lic_rows = "\n".join(
        f"| {lic} | {cnt:,} |"
        for lic, cnt in sorted(stats["by_license"].items(), key=lambda x: x[1], reverse=True)
    )

    return f"""---
language:
- th
- en
license:
- odc-by
- cc-by-4.0
task_categories:
- image-to-text
tags:
- ocr
- thai
- government-documents
- pdf
- benchmark
- data-go-th
pretty_name: Thai Government PDF OCR Test Benchmark
---

# Thai Government PDF OCR Test Benchmark (`narawit16/thai-govern-test-ocr`)

A diverse OCR evaluation benchmark of **Thai government documents** harvested from Thailand's Open Government Data Portal ([data.go.th](https://data.go.th)).

## Dataset Summary

- **Total Benchmark Pages (`test` split):** {stats['total_pages']:,} pages
- **Unique Source PDFs:** {stats['total_documents']:,} documents
- **Unique Government Organizations:** {stats['total_organizations']:,} agencies
- **Render Resolution:** {stats['dpi']} DPI (RGB PNG)

### Page Type Breakdown (`page_type`)
- **Scanned (`scanned` + `scanned_ocr`):** {stats['scanned_share_pct']}%
- **Complex Digital (`digital`):** {stats['digital_share_pct']}%

| `page_type` | Description | Pages |
|---|---|---:|
{cat_rows}

### Domain Group Breakdown (`group`)
| Group (`data.go.th`) | Pages |
|---|---:|
{group_rows}

### Source Licenses (`license_id`)
| License | Pages |
|---|---:|
{lic_rows}

## Schema

- `image`: Rendered page image (`PIL.Image`) at {stats['dpi']} DPI
- `doc_id`, `resource_id`: CKAN dataset and resource UUIDs on `data.go.th`
- `page_num`, `total_pages`: 1-based page number and total pages in the source PDF
- `page_type`: `scanned` | `scanned_ocr` | `digital`
- `image_coverage_ratio`: Fraction of page area covered by embedded raster images (`0.0` - `1.0`)
- `layout_complexity_score`, `path_count`, `text_rect_count`: Layout/table complexity metrics
- `has_embedded_text`, `embedded_text_chars`, `embedded_text_preview`: Embedded PDF text diagnostics (useful as weak reference or to detect broken Thai font encoding / hidden OCR layers)
- `group`, `organization`, `title`, `resource_name`, `license_id`, `source_url`, `metadata_modified`, `sha256`, `width`, `height`

## Usage

```python
from datasets import load_dataset

ds = load_dataset("narawit16/thai-govern-test-ocr", split="test")

# Evaluate separately on pure scanned vs complex digital pages
scanned_ds = ds.filter(lambda x: x["page_type"] == "scanned")
digital_ds = ds.filter(lambda x: x["page_type"] == "digital")

sample = scanned_ds[0]
sample["image"].show()
print(sample["organization"], sample["page_type"], sample["title"])
```
"""


def main() -> None:
    creds = load_credentials()
    if not HF_DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {HF_DATASET_DIR}")

    stats_path = DATA_DIR / "dataset_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    ds_dict = DatasetDict.load_from_disk(str(HF_DATASET_DIR))
    print(f"Pushing {stats['total_pages']} pages to {HF_REPO_ID}...")
    ds_dict.push_to_hub(
        HF_REPO_ID,
        token=creds.hf_token,
        commit_message=f"Upload Thai government OCR benchmark ({stats['total_pages']} pages from {stats['total_documents']} PDFs)",
    )

    readme_path = DATA_DIR / "README.md"
    readme_path.write_text(build_readme(stats), encoding="utf-8")

    api = HfApi(token=creds.hf_token)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message="Add Dataset Card README.md",
    )
    print(f"Successfully published to https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
