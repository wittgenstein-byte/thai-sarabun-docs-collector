"""Scan Saraban PDFs page-by-page: page_type, PUA check, PII regex, smart signature_page, header verify."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pypdfium2 as pdfium

from config import (
    MAX_SCAN_PAGES_PER_PDF,
    SARABAN_MAX_PAGES_PER_PDF,
    SARABAN_PAGE_INDEX_PATH,
    SARABAN_MANIFEST_PATH,
    SARABAN_SCANNED_RATIO,
    SARABAN_TARGET_PER_TYPE,
    SARABAN_TYPES,
)
from saraban_utils import (
    SIGNATURE_MARKERS,
    detect_header_saraban_type,
    has_pua_mojibake,
    has_thai_citizen_id,
)
from scan_pages import analyze_page, pick_scan_indices


def extract_full_text(page: pdfium.PdfPage) -> str:
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

    # Multi-page (>3): search scanned pages from index 1+, preferring earliest match in first 6 pages
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

    # Scanned multi-page docs without text layer: pick page 2 or 3 (index 1 or 2) rather than trailing annex
    for candidate in (2, 1):
        if candidate in page_texts:
            return candidate
    return None


def scan_saraban_all() -> None:
    if not SARABAN_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {SARABAN_MANIFEST_PATH}")

    manifest = [
        json.loads(line)
        for line in SARABAN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    existing_records: list[dict] = []
    scanned_res_ids: set[str] = set()
    if SARABAN_PAGE_INDEX_PATH.exists():
        for line in SARABAN_PAGE_INDEX_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            existing_records.append(rec)
            scanned_res_ids.add(rec["resource_id"])

    new_count = 0
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

                # Determine doc-level label_confidence from page 0 header
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

    print(f"Scanned {new_count} new pages (total indexed: {len(existing_records)} pages across {len(scanned_res_ids)} PDFs)")

    # Print usable saraban_type x page_type summary (capped at SARABAN_MAX_PAGES_PER_PDF per PDF)
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
    print(f"Target per type: {SARABAN_TARGET_PER_TYPE} ({target_scanned} scanned/scanned_ocr + {target_digital} digital)")
    print("\n--- Usable Pool by saraban_type (capped at 3/PDF) ---")
    for st in SARABAN_TYPES:
        sc = matrix[st]["scanned_pool"]
        dg = matrix[st]["digital"]
        print(f"  {st:18s}: scanned+ocr={sc:3d} (need {target_scanned}) | digital={dg:3d} (need {target_digital})")


if __name__ == "__main__":
    scan_saraban_all()
