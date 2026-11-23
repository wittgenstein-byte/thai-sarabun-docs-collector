"""Preprocessing pipeline for DLA bulk corpus: page scanning, PII screening, PNG rendering, and HF Dataset building."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from train_ocr.common.manifest import read_jsonl
from train_ocr.common.pdf_io import pick_scan_indices
from train_ocr.common.saraban_rules import (
    has_pua_mojibake,
    has_thai_citizen_id,
    is_pii_blocked_title,
)
from train_ocr.common.settings import (
    DLA_BULK_DIR,
    DLA_BULK_HF_DIR,
    DLA_BULK_MANIFEST_PATH,
    DLA_BULK_PAGE_INDEX_PATH,
    DLA_BULK_PAGES_DIR,
    RENDER_DPI,
)
from train_ocr.screener.pdf_scanner import analyze_page, extract_full_text


def scan_dla_bulk(
    manifest_path: Path = DLA_BULK_MANIFEST_PATH,
    page_index_path: Path = DLA_BULK_PAGE_INDEX_PATH,
    max_scan_pages: int = 10,
) -> list[dict]:
    """Scan DLA bulk PDFs page-by-page into page_index.jsonl (resumable)."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"DLA bulk manifest not found: {manifest_path}")

    manifest = read_jsonl(manifest_path)
    existing = read_jsonl(page_index_path)
    scanned_ids = {rec["doc_id"] for rec in existing}

    new_count = 0
    page_index_path.parent.mkdir(parents=True, exist_ok=True)
    with page_index_path.open("a", encoding="utf-8") as out_f:
        for item in manifest:
            doc_id = item["doc_id"]
            if doc_id in scanned_ids:
                continue

            if item.get("ext", "").lower() != ".pdf":
                continue

            pdf_path = Path(item["local_path"])
            if not pdf_path.exists():
                continue

            title_pii = is_pii_blocked_title(
                f"{item.get('title', '')} {item.get('link_label', '')} {item.get('original_filename', '')}"
            )

            doc = None
            try:
                doc = pdfium.PdfDocument(str(pdf_path))
                total_pages = len(doc)
                indices = pick_scan_indices(total_pages, max_scan_pages)
                for idx in indices:
                    page = doc[idx]
                    try:
                        metrics = analyze_page(page)
                        full_text = extract_full_text(page)
                    finally:
                        page.close()

                    row = {
                        "doc_id": doc_id,
                        "source": item["source"],
                        "page_index": idx,
                        "page_num": idx + 1,
                        "total_pages": total_pages,
                        "title_pii_flagged": title_pii,
                        "embedded_pii_flagged": has_thai_citizen_id(full_text),
                        "has_pua_mojibake": has_pua_mojibake(full_text),
                        **metrics,
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    existing.append(row)
                    new_count += 1
                scanned_ids.add(doc_id)
            except Exception as exc:
                print(f"  [warn] Error scanning {pdf_path.name}: {exc}")
            finally:
                if doc is not None:
                    doc.close()

    print(
        f"[scan] Scanned {new_count:,} new pages "
        f"(total indexed: {len(existing):,} pages across {len(scanned_ids):,} PDFs)"
    )
    return existing


def _round_robin_by_source(
    candidates: list[dict],
    target_total: int | None,
    max_pages_per_pdf: int,
) -> list[dict]:
    """Select pages balanced round-robin across DLA sources, respecting max_pages_per_pdf."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_source[row["source"]].append(row)

    source_order = sorted(by_source.keys(), key=lambda s: len(by_source[s]), reverse=True)
    pos = {s: 0 for s in source_order}
    used_per_doc: dict[str, int] = defaultdict(int)
    selected: list[dict] = []

    while source_order:
        if target_total is not None and len(selected) >= target_total:
            break
        next_sources = []
        for src in source_order:
            if target_total is not None and len(selected) >= target_total:
                break
            items = by_source[src]
            picked = False
            while pos[src] < len(items):
                cand = items[pos[src]]
                pos[src] += 1
                if used_per_doc[cand["doc_id"]] < max_pages_per_pdf:
                    used_per_doc[cand["doc_id"]] += 1
                    selected.append(cand)
                    picked = True
                    break
            if picked and pos[src] < len(items):
                next_sources.append(src)
        source_order = next_sources

    return selected


def build_dla_bulk_dataset(
    manifest_path: Path = DLA_BULK_MANIFEST_PATH,
    page_index_path: Path = DLA_BULK_PAGE_INDEX_PATH,
    pages_dir: Path = DLA_BULK_PAGES_DIR,
    hf_dir: Path = DLA_BULK_HF_DIR,
    *,
    dpi: int = RENDER_DPI,
    max_pages_per_pdf: int = 3,
    target_total: int | None = 2000,
    split: str = "train",
    skip_pii: bool = True,
) -> dict[str, Any]:
    """Render selected DLA bulk pages to RGB PNG and build a Hugging Face DatasetDict."""
    from datasets import Dataset, DatasetDict, Features, Image, Value

    manifest_map = {rec["doc_id"]: rec for rec in read_jsonl(manifest_path)}
    indexed = read_jsonl(page_index_path)
    if not indexed:
        raise RuntimeError("No scanned pages found in page_index.jsonl. Run scan first.")

    pii_excluded = 0
    other_excluded = 0
    pool: list[dict] = []
    for row in indexed:
        if skip_pii and (row.get("title_pii_flagged") or row.get("embedded_pii_flagged")):
            pii_excluded += 1
            continue
        if row.get("page_type") == "other":
            other_excluded += 1
            continue
        pool.append(row)

    # Prefer scanned & complex digital pages first
    priority = {"scanned": 0, "scanned_ocr": 1, "digital": 2}
    pool.sort(
        key=lambda r: (
            priority.get(r.get("page_type", "digital"), 3),
            -float(r.get("layout_complexity_score", 0.0)),
        )
    )

    selected = _round_robin_by_source(pool, target_total=target_total, max_pages_per_pdf=max_pages_per_pdf)
    if not selected:
        raise RuntimeError("No eligible pages left after filtering.")

    pages_by_doc: dict[str, list[dict]] = defaultdict(list)
    for pm in selected:
        pages_by_doc[pm["doc_id"]].append(pm)

    pages_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    rows: list[dict[str, Any]] = []

    for doc_id, page_metas in pages_by_doc.items():
        rec = manifest_map.get(doc_id)
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
                if idx >= len(doc):
                    continue
                page = doc[idx]
                try:
                    bitmap = page.render(scale=scale)
                    pil_img = bitmap.to_pil().convert("RGB")
                    width, height = pil_img.size
                finally:
                    page.close()

                out_dir = pages_dir / rec["source"]
                out_dir.mkdir(parents=True, exist_ok=True)
                img_path = out_dir / f"{doc_id}_p{pm['page_num']}.png"
                pil_img.save(str(img_path), format="PNG")

                rows.append(
                    {
                        "image": str(img_path),
                        "doc_id": doc_id,
                        "source": rec["source"],
                        "subkey": rec.get("subkey", ""),
                        "dept": rec.get("dept", ""),
                        "doc_no": rec.get("doc_no", ""),
                        "doc_date": rec.get("doc_date", ""),
                        "title": rec.get("title", ""),
                        "page_num": int(pm["page_num"]),
                        "total_pages": int(pm["total_pages"]),
                        "page_type": pm["page_type"],
                        "image_coverage_ratio": float(pm["image_coverage_ratio"]),
                        "embedded_text_chars": int(pm["embedded_text_chars"]),
                        "embedded_text_preview": pm.get("embedded_text_preview", ""),
                        "layout_complexity_score": float(pm["layout_complexity_score"]),
                        "has_pua_mojibake": bool(pm.get("has_pua_mojibake", False)),
                        "dpi": int(dpi),
                        "width": int(width),
                        "height": int(height),
                        "source_url": rec.get("source_url", ""),
                        "sha256": rec.get("sha256", ""),
                    }
                )
        except Exception as exc:
            print(f"  [warn] Failed rendering {pdf_path.name}: {exc}")
        finally:
            if doc is not None:
                doc.close()

    features = Features(
        {
            "image": Image(),
            "doc_id": Value("string"),
            "source": Value("string"),
            "subkey": Value("string"),
            "dept": Value("string"),
            "doc_no": Value("string"),
            "doc_date": Value("string"),
            "title": Value("string"),
            "page_num": Value("int32"),
            "total_pages": Value("int32"),
            "page_type": Value("string"),
            "image_coverage_ratio": Value("float32"),
            "embedded_text_chars": Value("int32"),
            "embedded_text_preview": Value("string"),
            "layout_complexity_score": Value("float32"),
            "has_pua_mojibake": Value("bool"),
            "dpi": Value("int32"),
            "width": Value("int32"),
            "height": Value("int32"),
            "source_url": Value("string"),
            "sha256": Value("string"),
        }
    )

    ds = Dataset.from_list(rows, features=features)
    ds_dict = DatasetDict({split: ds})
    hf_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(hf_dir))

    by_source = dict(Counter(r["source"] for r in rows))
    by_page_type = dict(Counter(r["page_type"] for r in rows))
    scanned_count = by_page_type.get("scanned", 0) + by_page_type.get("scanned_ocr", 0)
    digital_count = by_page_type.get("digital", 0)
    total_pages = len(rows)

    stats = {
        "config_name": "dla_bulk",
        "split": split,
        "total_pages": total_pages,
        "total_documents": len({r["doc_id"] for r in rows}),
        "dpi": dpi,
        "pii_excluded_pages": pii_excluded,
        "other_excluded_pages": other_excluded,
        "scanned_share_pct": round(100.0 * scanned_count / max(total_pages, 1), 1),
        "digital_share_pct": round(100.0 * digital_count / max(total_pages, 1), 1),
        "by_source": by_source,
        "by_page_type": by_page_type,
    }
    stats_path = hf_dir.parent / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[build] Saved {total_pages:,} pages ({stats['total_documents']:,} PDFs, {dpi} DPI) "
        f"to {hf_dir} (pii_excluded={pii_excluded}, other_excluded={other_excluded})"
    )
    print(f"[build] By source: {by_source}")
    print(f"[build] By page_type: {by_page_type}")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess DLA bulk PDFs into a Hugging Face dataset.")
    parser.add_argument("--data-dir", type=Path, default=DLA_BULK_DIR)
    parser.add_argument("--scan-pages-per-pdf", type=int, default=10)
    parser.add_argument("--max-pages-per-pdf", type=int, default=3)
    parser.add_argument("--target-total", type=int, default=2000, help="Max total pages to render (0 = all)")
    parser.add_argument("--dpi", type=int, default=RENDER_DPI)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--keep-pii", action="store_true", help="Do not filter PII-flagged pages")
    parser.add_argument("--scan-only", action="store_true", help="Only scan PDFs into page_index.jsonl")
    args = parser.parse_args(argv)

    manifest_path = args.data_dir / "manifest.jsonl"
    page_index_path = args.data_dir / "page_index.jsonl"
    pages_dir = args.data_dir / "pages"
    hf_dir = args.data_dir / "hf_dataset"

    scan_dla_bulk(
        manifest_path=manifest_path,
        page_index_path=page_index_path,
        max_scan_pages=args.scan_pages_per_pdf,
    )
    if not args.scan_only:
        build_dla_bulk_dataset(
            manifest_path=manifest_path,
            page_index_path=page_index_path,
            pages_dir=pages_dir,
            hf_dir=hf_dir,
            dpi=args.dpi,
            max_pages_per_pdf=args.max_pages_per_pdf,
            target_total=None if args.target_total <= 0 else args.target_total,
            split=args.split,
            skip_pii=not args.keep_pii,
        )
