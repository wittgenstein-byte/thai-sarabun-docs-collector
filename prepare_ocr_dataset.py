"""Stratified sampling (75% scanned/scanned_ocr, 25% complex digital) and HF Dataset builder."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pypdfium2 as pdfium
from datasets import Dataset, DatasetDict, Features, Image, Value

from config import (
    DATA_DIR,
    GROUPS,
    HF_DATASET_DIR,
    MANIFEST_PATH,
    MAX_PAGES_PER_PDF,
    PAGE_INDEX_PATH,
    PAGES_DIR,
    RENDER_DPI,
    SCANNED_RATIO,
    TARGET_TOTAL_PAGES,
)


def round_robin_select(
    candidates: list[dict],
    target_count: int,
    max_per_pdf: int,
    already_used_per_pdf: dict[str, int],
) -> list[dict]:
    """Select up to target_count pages balanced across groups, respecting max_per_pdf."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for item in candidates:
        by_group[item["group"]].append(item)

    selected: list[dict] = []
    group_order = [g for g in GROUPS if by_group.get(g)]
    pos = {g: 0 for g in group_order}

    while len(selected) < target_count and group_order:
        next_groups = []
        for g in group_order:
            if len(selected) >= target_count:
                break
            items = by_group[g]
            picked = False
            while pos[g] < len(items):
                cand = items[pos[g]]
                pos[g] += 1
                res_id = cand["resource_id"]
                if already_used_per_pdf[res_id] < max_per_pdf:
                    already_used_per_pdf[res_id] += 1
                    selected.append(cand)
                    picked = True
                    break
            if picked and pos[g] < len(items):
                next_groups.append(g)
        group_order = next_groups

    return selected


def build_dataset(
    target_total: int = TARGET_TOTAL_PAGES,
    scanned_ratio: float = SCANNED_RATIO,
    max_pages_per_pdf: int = MAX_PAGES_PER_PDF,
    dpi: int = RENDER_DPI,
) -> None:
    if not MANIFEST_PATH.exists() or not PAGE_INDEX_PATH.exists():
        raise FileNotFoundError("Missing data/manifest.jsonl or data/page_index.jsonl. Run scan_pages.py first.")

    manifest_map = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            manifest_map[rec["resource_id"]] = rec

    indexed_pages = [
        json.loads(line)
        for line in PAGE_INDEX_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Separate pools: prioritize pure 'scanned', then 'scanned_ocr'
    scanned_pool = [p for p in indexed_pages if p["page_type"] == "scanned"]
    scanned_ocr_pool = [p for p in indexed_pages if p["page_type"] == "scanned_ocr"]
    # Sort digital pages by descending layout_complexity_score (complex tables/multi-column first)
    digital_pool = sorted(
        (p for p in indexed_pages if p["page_type"] == "digital"),
        key=lambda x: (x["layout_complexity_score"], x["path_count"], x["text_rect_count"]),
        reverse=True,
    )

    used_per_pdf: dict[str, int] = defaultdict(int)
    target_scanned = round(target_total * scanned_ratio)

    selected_scanned = round_robin_select(scanned_pool, target_scanned, max_pages_per_pdf, used_per_pdf)
    if len(selected_scanned) < target_scanned:
        rem = target_scanned - len(selected_scanned)
        selected_scanned.extend(
            round_robin_select(scanned_ocr_pool, rem, max_pages_per_pdf, used_per_pdf)
        )

    # Guarantee exact ratio (70-80% scanned / 20-30% digital) even if scanned pool is smaller
    target_digital = round(len(selected_scanned) * (1.0 - scanned_ratio) / scanned_ratio)
    selected_digital = round_robin_select(digital_pool, target_digital, max_pages_per_pdf, used_per_pdf)

    selected_pages = selected_scanned + selected_digital
    if not selected_pages:
        raise RuntimeError("No pages selected for benchmark.")

    # Group selected pages by resource_id so each PDF is opened once
    pages_by_pdf: dict[str, list[dict]] = defaultdict(list)
    for p in selected_pages:
        pages_by_pdf[p["resource_id"]].append(p)

    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    scale = dpi / 72.0

    for res_id, page_metas in pages_by_pdf.items():
        rec = manifest_map.get(res_id)
        if not rec:
            continue
        pdf_path = Path(rec["local_path"])
        if not pdf_path.exists():
            continue

        doc = None
        try:
            doc = pdfium.PdfDocument(str(pdf_path))
            for pm in sorted(page_metas, key=lambda x: x["page_index"]):
                idx = pm["page_index"]
                page = doc[idx]
                try:
                    bitmap = page.render(scale=scale)
                    pil_img = bitmap.to_pil().convert("RGB")
                    width, height = pil_img.size
                finally:
                    page.close()

                out_dir = PAGES_DIR / rec["group"]
                out_dir.mkdir(parents=True, exist_ok=True)
                img_path = out_dir / f"{res_id}_p{pm['page_num']}.png"
                pil_img.save(str(img_path), format="PNG")

                rows.append(
                    {
                        "image": str(img_path),
                        "doc_id": rec["doc_id"],
                        "resource_id": res_id,
                        "page_num": pm["page_num"],
                        "total_pages": rec["total_pages"],
                        "page_type": pm["page_type"],
                        "image_coverage_ratio": float(pm["image_coverage_ratio"]),
                        "layout_complexity_score": float(pm["layout_complexity_score"]),
                        "path_count": int(pm["path_count"]),
                        "text_rect_count": int(pm["text_rect_count"]),
                        "has_embedded_text": pm["embedded_text_chars"] >= 20,
                        "embedded_text_chars": int(pm["embedded_text_chars"]),
                        "embedded_text_preview": pm["embedded_text_preview"],
                        "group": rec["group"],
                        "organization": rec["organization"],
                        "title": rec["title"],
                        "resource_name": rec["resource_name"],
                        "license_id": rec["license_id"],
                        "source_url": rec["source_url"],
                        "metadata_modified": rec["metadata_modified"],
                        "sha256": rec["sha256"],
                        "width": width,
                        "height": height,
                    }
                )
        except Exception as exc:
            print(f"Skipping {pdf_path.name} due to render error: {exc}")
        finally:
            if doc is not None:
                doc.close()

    features = Features(
        {
            "image": Image(),
            "doc_id": Value("string"),
            "resource_id": Value("string"),
            "page_num": Value("int32"),
            "total_pages": Value("int32"),
            "page_type": Value("string"),
            "image_coverage_ratio": Value("float32"),
            "layout_complexity_score": Value("float32"),
            "path_count": Value("int32"),
            "text_rect_count": Value("int32"),
            "has_embedded_text": Value("bool"),
            "embedded_text_chars": Value("int32"),
            "embedded_text_preview": Value("string"),
            "group": Value("string"),
            "organization": Value("string"),
            "title": Value("string"),
            "resource_name": Value("string"),
            "license_id": Value("string"),
            "source_url": Value("string"),
            "metadata_modified": Value("string"),
            "sha256": Value("string"),
            "width": Value("int32"),
            "height": Value("int32"),
        }
    )

    ds = Dataset.from_list(rows, features=features)
    ds_dict = DatasetDict({"test": ds})
    ds_dict.save_to_disk(str(HF_DATASET_DIR))

    type_counts = dict(Counter(r["page_type"] for r in rows))
    scanned_total = type_counts.get("scanned", 0) + type_counts.get("scanned_ocr", 0)
    stats = {
        "total_pages": len(rows),
        "total_documents": len({r["resource_id"] for r in rows}),
        "total_organizations": len({r["organization"] for r in rows}),
        "by_page_type": type_counts,
        "scanned_share_pct": round(100.0 * scanned_total / len(rows), 2),
        "digital_share_pct": round(100.0 * type_counts.get("digital", 0) / len(rows), 2),
        "by_group": dict(Counter(r["group"] for r in rows)),
        "by_license": dict(Counter(r["license_id"] for r in rows)),
        "dpi": dpi,
    }
    (DATA_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built HF Dataset: {stats['total_pages']} pages from {stats['total_documents']} PDFs")
    print(f"page_type counts: {stats['by_page_type']} (scanned={stats['scanned_share_pct']}%, digital={stats['digital_share_pct']}%)")
    print(f"Organizations: {stats['total_organizations']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified OCR benchmark dataset.")
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL_PAGES)
    parser.add_argument("--scanned-ratio", type=float, default=SCANNED_RATIO)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_PDF)
    parser.add_argument("--dpi", type=int, default=RENDER_DPI)
    args = parser.parse_args()
    build_dataset(
        target_total=args.target_total,
        scanned_ratio=args.scanned_ratio,
        max_pages_per_pdf=args.max_pages,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
