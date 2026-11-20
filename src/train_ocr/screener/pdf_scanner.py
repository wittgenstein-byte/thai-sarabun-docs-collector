"""Page-by-page PDF scanner for general data.go.th and 6-type Thai Saraban corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from train_ocr.common.manifest import read_jsonl
from train_ocr.common.pdf_io import compute_image_coverage, pick_scan_indices
from train_ocr.common.saraban_rules import (
    SIGNATURE_MARKERS,
    detect_header_saraban_type,
    has_pua_mojibake,
    has_thai_citizen_id,
)
from train_ocr.common.settings import (
    IMAGE_COVERAGE_THRESHOLD,
    MANIFEST_PATH,
    MAX_PAGES_PER_PDF,
    MAX_SCAN_PAGES_PER_PDF,
    PAGE_INDEX_PATH,
    SARABAN_MANIFEST_PATH,
    SARABAN_MAX_PAGES_PER_PDF,
    SARABAN_PAGE_INDEX_PATH,
    SARABAN_SCANNED_RATIO,
    SARABAN_TARGET_PER_TYPE,
    SARABAN_TYPES,
    SCANNED_RATIO,
    TARGET_TOTAL_PAGES,
)


def analyze_page(page: Any) -> dict[str, Any]:
    """Analyze a single PDF page and return classification and layout metrics."""
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


def extract_full_text(page: Any) -> str:
    """Extract full embedded text for PII and signature-marker detection."""
    tp = page.get_textpage()
    try:
        return (tp.get_text_range() or "").strip()
    finally:
        tp.close()


def find_signature_page_idx(total_pages: int, page_texts: dict[int, str]) -> int | None:
    """Determine the true signature page index, avoiding appendix/attachment traps."""
    if total_pages <= 1:
        return None
    if total_pages <= 3:
        return total_pages - 1

    early_matches = []
    late_matches = []
    for idx in sorted(page_texts):
        if idx == 0:
            continue
        txt = page_texts[idx]
        if any(m in txt for m in SIGNATURE_MARKERS):
            if idx < 6:
                early_matches.append(idx)
            else:
                late_matches.append(idx)

    if early_matches:
        return early_matches[0]
    if late_matches:
        return late_matches[0]

    for candidate in (2, 1):
        if candidate in page_texts:
            return candidate
    return None


def scan_general_all() -> None:
    """Scan general data.go.th PDFs page-by-page into data/page_index.jsonl."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    manifest = read_jsonl(MANIFEST_PATH)
    existing_records = read_jsonl(PAGE_INDEX_PATH)
    scanned_res_ids = {rec["resource_id"] for rec in existing_records}

    new_count = 0
    PAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
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

    print(
        f"Scanned {new_count} new pages "
        f"(total indexed: {len(existing_records)} pages across {len(scanned_res_ids)} PDFs)"
    )

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
    print(
        f"Target scanned+scanned_ocr pages ({int(SCANNED_RATIO * 100)}% of {TARGET_TOTAL_PAGES}): "
        f"{target_scanned_pages}"
    )
    gap = max(0, target_scanned_pages - usable_scanned_pages)
    print(f"Gap to fill: {gap} scanned pages (~{gap // 3} more scanned PDFs)")
    print("\nUsable scanned pages by group:", dict(sorted(scanned_cap_per_group.items())))


def scan_saraban_all() -> None:
    """Scan Saraban PDFs page-by-page into data/saraban/page_index.jsonl."""
    if not SARABAN_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {SARABAN_MANIFEST_PATH}")

    manifest = read_jsonl(SARABAN_MANIFEST_PATH)
    existing_records = read_jsonl(SARABAN_PAGE_INDEX_PATH)
    scanned_res_ids = {rec["resource_id"] for rec in existing_records}

    new_count = 0
    SARABAN_PAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SARABAN_PAGE_INDEX_PATH.open("a", encoding="utf-8") as out_f:
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
                total_doc_pages = len(doc)
                indices = pick_scan_indices(total_doc_pages, MAX_SCAN_PAGES_PER_PDF)
                if 0 not in indices:
                    indices = [0] + indices

                page_metrics: dict[int, dict] = {}
                page_texts: dict[int, str] = {}
                for idx in indices:
                    page = doc[idx]
                    try:
                        metrics = analyze_page(page)
                        full_text = extract_full_text(page)
                    finally:
                        page.close()
                    page_metrics[idx] = metrics
                    page_texts[idx] = full_text

                p0_text = page_texts.get(0, "")
                p0_pua = has_pua_mojibake(p0_text)
                if len(p0_text) >= 20 and not p0_pua:
                    detected = detect_header_saraban_type(p0_text)
                    if detected == item["saraban_type"]:
                        label_conf = "high"
                    elif detected is not None:
                        label_conf = "mismatch_needs_review"
                    else:
                        label_conf = "needs_visual_verify"
                else:
                    label_conf = "needs_visual_verify"

                sig_idx = find_signature_page_idx(total_doc_pages, page_texts)

                for idx in indices:
                    if idx == 0:
                        role = "first_page_header"
                    elif sig_idx is not None and idx == sig_idx:
                        role = "signature_page"
                    else:
                        role = "body_table"

                    full_text = page_texts[idx]
                    row = {
                        "resource_id": res_id,
                        "doc_id": item["doc_id"],
                        "saraban_type": item["saraban_type"],
                        "saraban_subtype": item["saraban_subtype"],
                        "meta_doc_no": item["meta_doc_no"],
                        "meta_doc_date": item["meta_doc_date"],
                        "meta_subject": item["meta_subject"],
                        "organization": item["organization"],
                        "source_portal": item["source_portal"],
                        "page_index": idx,
                        "page_num": idx + 1,
                        "total_pages": total_doc_pages,
                        "page_role": role,
                        "label_confidence": label_conf,
                        "has_pua_mojibake": has_pua_mojibake(full_text),
                        "embedded_pii_flagged": has_thai_citizen_id(full_text),
                        **page_metrics[idx],
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

    print(
        f"Scanned {new_count} new pages "
        f"(total indexed: {len(existing_records)} pages across {len(scanned_res_ids)} PDFs)"
    )

    per_pdf_type: dict[tuple[str, str, str], int] = defaultdict(int)
    pii_count = 0
    mismatch_count = 0
    for r in existing_records:
        if r.get("embedded_pii_flagged"):
            pii_count += 1
            continue
        if r.get("label_confidence") == "mismatch_needs_review":
            mismatch_count += 1
            continue
        if r["page_type"] == "other":
            continue
        bucket = "scanned_pool" if r["page_type"] in {"scanned", "scanned_ocr"} else "digital"
        key = (r["saraban_type"], bucket, r["resource_id"])
        if per_pdf_type[key] < SARABAN_MAX_PAGES_PER_PDF:
            per_pdf_type[key] += 1

    matrix: dict[str, Counter] = {t: Counter() for t in SARABAN_TYPES}
    for (st, bucket, _), cnt in per_pdf_type.items():
        matrix.setdefault(st, Counter())[bucket] += cnt

    target_scanned = round(SARABAN_TARGET_PER_TYPE * SARABAN_SCANNED_RATIO)
    target_digital = SARABAN_TARGET_PER_TYPE - target_scanned
    print(f"\nExcluded: embedded_pii_flagged={pii_count} pages, mismatch_needs_review={mismatch_count} pages")
    print(
        f"Target per type: {SARABAN_TARGET_PER_TYPE} "
        f"({target_scanned} scanned/scanned_ocr + {target_digital} digital)"
    )
    print("\n--- Usable Pool by saraban_type (capped at 3/PDF) ---")
    for st in SARABAN_TYPES:
        sc = matrix[st]["scanned_pool"]
        dg = matrix[st]["digital"]
        print(
            f"  {st:18s}: scanned+ocr={sc:3d} (need {target_scanned}) | digital={dg:3d} (need {target_digital})"
        )


def main_general(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan general data.go.th PDFs page-by-page.")
    parser.parse_args(argv)
    scan_general_all()


def main_saraban(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan Saraban PDFs page-by-page.")
    parser.parse_args(argv)
    scan_saraban_all()
