"""Scan downloaded PDFs page-by-page to classify scanned, scanned_ocr, and digital pages."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pypdfium2 as pdfium

from config import (
    IMAGE_COVERAGE_THRESHOLD,
    MANIFEST_PATH,
    MAX_PAGES_PER_PDF,
    MAX_SCAN_PAGES_PER_PDF,
    PAGE_INDEX_PATH,
    SCANNED_RATIO,
    TARGET_TOTAL_PAGES,
)


def compute_image_coverage(page: pdfium.PdfPage) -> float:
    """Compute fraction of page area covered by embedded raster images (0.0 - 1.0)."""
    w, h = page.get_size()
    page_area = max(w * h, 1.0)
    img_area = 0.0
    try:
        for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
            l, b, r, t = obj.get_bounds()
            cw = max(0.0, min(r, w) - max(l, 0.0))
            ch = max(0.0, min(t, h) - max(b, 0.0))
            img_area += cw * ch
    except Exception:
        return 0.0
    return min(1.0, img_area / page_area)


def analyze_page(page: pdfium.PdfPage) -> dict:
    """Analyze a single page and return classification and layout metrics."""
    img_cov = compute_image_coverage(page)
    try:
        path_count = sum(1 for _ in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_PATH]))
    except Exception:
        path_count = 0

    tp = page.get_textpage()
    try:
        text = (tp.get_text_range() or "").strip()
        text_rect_count = tp.count_rects()
    finally:
        tp.close()

    char_count = len(text)
    if char_count < 20 and img_cov > IMAGE_COVERAGE_THRESHOLD:
        page_type = "scanned"
    elif char_count >= 20 and img_cov > IMAGE_COVERAGE_THRESHOLD:
        page_type = "scanned_ocr"
    elif char_count >= 50 and img_cov <= IMAGE_COVERAGE_THRESHOLD:
        page_type = "digital"
    else:
        page_type = "other"

    digits = sum(1 for c in text if c.isdigit())
    digit_ratio = digits / max(char_count, 1)
    complexity = (
        min(path_count, 300) / 300.0 * 0.5
        + min(text_rect_count, 150) / 150.0 * 0.3
        + digit_ratio * 0.2
    )

    return {
        "page_type": page_type,
        "image_coverage_ratio": round(img_cov, 4),
        "embedded_text_chars": char_count,
        "embedded_text_preview": " ".join(text.split())[:300],
        "path_count": path_count,
        "text_rect_count": text_rect_count,
        "layout_complexity_score": round(complexity, 4),
    }


def pick_scan_indices(total_pages: int, max_scan: int) -> list[int]:
    """Select up to max_scan evenly spaced 0-based page indices."""
    if total_pages <= max_scan:
        return list(range(total_pages))
    step = (total_pages - 1) / (max_scan - 1)
    return sorted({min(total_pages - 1, round(i * step)) for i in range(max_scan)})


def scan_all() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = [
        json.loads(line)
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    existing_records: list[dict] = []
    scanned_res_ids: set[str] = set()
    if PAGE_INDEX_PATH.exists():
        for line in PAGE_INDEX_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            existing_records.append(rec)
            scanned_res_ids.add(rec["resource_id"])

    new_count = 0
    with PAGE_INDEX_PATH.open("a", encoding="utf-8") as out_f:
        for item in manifest:
            res_id = item["resource_id"]
            if res_id in scanned_res_ids:
                continue

            pdf_path = Path(item["local_path"])
            if not pdf_path.exists():
                continue

            doc = None
            try:
                doc = pdfium.PdfDocument(str(pdf_path))
                indices = pick_scan_indices(len(doc), MAX_SCAN_PAGES_PER_PDF)
                for idx in indices:
                    page = doc[idx]
                    try:
                        metrics = analyze_page(page)
                    finally:
                        page.close()

                    row = {
                        "resource_id": res_id,
                        "doc_id": item["doc_id"],
                        "group": item["group"],
                        "organization": item["organization"],
                        "page_index": idx,
                        "page_num": idx + 1,
                        "total_pages": item["total_pages"],
                        **metrics,
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    existing_records.append(row)
                    new_count += 1
                scanned_res_ids.add(res_id)
            except Exception as exc:
                print(f"Error scanning {pdf_path.name}: {exc}")
            finally:
                if doc is not None:
                    doc.close()

    print(f"Scanned {new_count} new pages (total indexed: {len(existing_records)} pages across {len(scanned_res_ids)} PDFs)")

    by_type = Counter(r["page_type"] for r in existing_records)
    pdfs_with_type: dict[str, set[str]] = defaultdict(set)
    scanned_cap_per_group: dict[str, int] = defaultdict(int)
    per_pdf_scanned: dict[str, tuple[str, int]] = {}

    for r in existing_records:
        pdfs_with_type[r["page_type"]].add(r["resource_id"])
        if r["page_type"] in {"scanned", "scanned_ocr"}:
            grp, cnt = per_pdf_scanned.get(r["resource_id"], (r["group"], 0))
            per_pdf_scanned[r["resource_id"]] = (grp, cnt + 1)

    for grp, cnt in per_pdf_scanned.values():
        scanned_cap_per_group[grp] += min(cnt, MAX_PAGES_PER_PDF)

    print("\n--- Page Index Summary ---")
    for ptype in ("scanned", "scanned_ocr", "digital", "other"):
        print(f"  {ptype:12s}: {by_type[ptype]:5d} pages ({len(pdfs_with_type[ptype]):3d} PDFs)")

    target_scanned_pages = int(TARGET_TOTAL_PAGES * SCANNED_RATIO)
    usable_scanned_pages = sum(scanned_cap_per_group.values())
    print(f"\nUsable scanned+scanned_ocr pages (capped at {MAX_PAGES_PER_PDF}/PDF): {usable_scanned_pages}")
    print(f"Target scanned+scanned_ocr pages ({int(SCANNED_RATIO * 100)}% of {TARGET_TOTAL_PAGES}): {target_scanned_pages}")
    gap = max(0, target_scanned_pages - usable_scanned_pages)
    print(f"Gap to fill: {gap} scanned pages (~{gap // 3} more scanned PDFs)")
    print("\nUsable scanned pages by group:", dict(sorted(scanned_cap_per_group.items())))


if __name__ == "__main__":
    scan_all()
