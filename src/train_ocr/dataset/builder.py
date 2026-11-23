"""Stratified OCR benchmark dataset builders (general data.go.th and 6-type Saraban)."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from PIL import Image as PILImage, ImageOps

from train_ocr.common.manifest import read_jsonl
from train_ocr.common.saraban_rules import detect_quality_tags
from train_ocr.common.settings import (
    DATA_DIR,
    GOOGLE_CRAWL_DIR,
    GOOGLE_HF_DIR,
    GOOGLE_PREPROCESSED_DIR,
    GROUPS,
    HF_DATASET_DIR,
    MANIFEST_PATH,
    MAX_PAGES_PER_PDF,
    PAGE_INDEX_PATH,
    PAGES_DIR,
    RENDER_DPI,
    SARABAN_DIR,
    SARABAN_HF_DIR,
    SARABAN_MANIFEST_PATH,
    SARABAN_MAX_PAGES_PER_PDF,
    SARABAN_PAGE_INDEX_PATH,
    SARABAN_PAGES_DIR,
    SARABAN_RENDER_DPI,
    SARABAN_SCANNED_RATIO,
    SARABAN_TARGET_PER_TYPE,
    SARABAN_TYPES,
    SCANNED_RATIO,
    TARGET_TOTAL_PAGES,
)
from train_ocr.screener.client import ScreenerClient

ROLE_PRIORITY = {"first_page_header": 0, "signature_page": 1, "body_table": 2}


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


def build_general_dataset(
    target_total: int = TARGET_TOTAL_PAGES,
    scanned_ratio: float = SCANNED_RATIO,
    max_pages_per_pdf: int = MAX_PAGES_PER_PDF,
    dpi: int = RENDER_DPI,
) -> None:
    """Build stratified general OCR benchmark dataset and save to data/hf_dataset."""
    from datasets import Dataset, DatasetDict, Features, Image, Value

    if not MANIFEST_PATH.exists() or not PAGE_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Missing data/manifest.jsonl or data/page_index.jsonl. Run scan-general first."
        )

    manifest_map = {rec["resource_id"]: rec for rec in read_jsonl(MANIFEST_PATH)}
    indexed_pages = read_jsonl(PAGE_INDEX_PATH)

    scanned_pool = [p for p in indexed_pages if p["page_type"] == "scanned"]
    scanned_ocr_pool = [p for p in indexed_pages if p["page_type"] == "scanned_ocr"]
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

    target_digital = round(len(selected_scanned) * (1.0 - scanned_ratio) / scanned_ratio)
    selected_digital = round_robin_select(digital_pool, target_digital, max_pages_per_pdf, used_per_pdf)

    selected_pages = selected_scanned + selected_digital
    if not selected_pages:
        raise RuntimeError("No pages selected for benchmark.")

    pages_by_pdf: dict[str, list[dict]] = defaultdict(list)
    for p in selected_pages:
        pages_by_pdf[p["resource_id"]].append(p)

    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
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
    print(
        f"page_type counts: {stats['by_page_type']} "
        f"(scanned={stats['scanned_share_pct']}%, digital={stats['digital_share_pct']}%)"
    )
    print(f"Organizations: {stats['total_organizations']}")


def order_candidates_diverse(candidates: list[dict], sort_by_complexity: bool = False) -> list[dict]:
    """Order candidates across PDFs round-robin while prioritizing first_page_header and signature_page."""
    by_pdf: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_pdf[c["resource_id"]].append(c)

    for items in by_pdf.values():
        if sort_by_complexity:
            items.sort(
                key=lambda x: (
                    ROLE_PRIORITY.get(x["page_role"], 9),
                    -x["layout_complexity_score"],
                )
            )
        else:
            items.sort(key=lambda x: (ROLE_PRIORITY.get(x["page_role"], 9), x["page_index"]))

    pdf_ids = list(by_pdf.keys())
    if sort_by_complexity:
        pdf_ids.sort(
            key=lambda rid: max(x["layout_complexity_score"] for x in by_pdf[rid]),
            reverse=True,
        )

    ordered: list[dict] = []
    idx_map = {rid: 0 for rid in pdf_ids}
    active = list(pdf_ids)
    while active:
        next_active = []
        for rid in active:
            i = idx_map[rid]
            if i < len(by_pdf[rid]):
                ordered.append(by_pdf[rid][i])
                idx_map[rid] = i + 1
                if idx_map[rid] < len(by_pdf[rid]):
                    next_active.append(rid)
        active = next_active
    return ordered


def build_saraban_dataset(
    per_type: int = SARABAN_TARGET_PER_TYPE,
    scanned_ratio: float = SARABAN_SCANNED_RATIO,
    max_pages_per_pdf: int = SARABAN_MAX_PAGES_PER_PDF,
    dpi: int = SARABAN_RENDER_DPI,
    screener_url: str | None = None,
) -> None:
    """Build 300 DPI stratified 6-type Thai Saraban OCR benchmark via ScreenerClient."""
    from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value

    if not SARABAN_MANIFEST_PATH.exists() or not SARABAN_PAGE_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Missing Saraban manifest or page_index. Run scan-saraban first."
        )

    manifest_map = {rec["resource_id"]: rec for rec in read_jsonl(SARABAN_MANIFEST_PATH)}
    indexed_pages = [
        rec
        for rec in read_jsonl(SARABAN_PAGE_INDEX_PATH)
        if not rec.get("embedded_pii_flagged", False)
        and rec.get("page_type") != "other"
        and rec.get("label_confidence") != "mismatch_needs_review"
    ]

    screener = ScreenerClient(base_url=screener_url)
    SARABAN_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0

    rows: list[dict[str, Any]] = []
    ocr_pii_rejected = 0

    for st in SARABAN_TYPES:
        pool = [p for p in indexed_pages if p["saraban_type"] == st]
        if st == "official_record":
            pool = [
                p
                for p in pool
                if p.get("label_confidence") == "high"
                or any(
                    kw
                    in (
                        p.get("meta_subject", "")
                        + " "
                        + manifest_map.get(p["resource_id"], {}).get("link_label", "")
                    )
                    for kw in (
                        "รายงานการประชุม",
                        "สรุปข้อราชการ",
                        "หนังสือรับรอง",
                        "มติ",
                        "บันทึกการประชุม",
                    )
                )
            ]

        scanned_pool = order_candidates_diverse([p for p in pool if p["page_type"] == "scanned"])
        scanned_ocr_pool = order_candidates_diverse(
            [p for p in pool if p["page_type"] == "scanned_ocr"]
        )
        digital_pool = order_candidates_diverse(
            [p for p in pool if p["page_type"] == "digital"],
            sort_by_complexity=True,
        )

        used_per_pdf: dict[str, int] = defaultdict(int)
        st_rows: list[dict[str, Any]] = []

        def try_render_and_add(cand: dict) -> bool:
            nonlocal ocr_pii_rejected
            res_id = cand["resource_id"]
            if used_per_pdf[res_id] >= max_pages_per_pdf:
                return False
            rec = manifest_map.get(res_id)
            if not rec:
                return False
            pdf_path = Path(rec["local_path"])
            if not pdf_path.exists():
                return False

            doc = None
            try:
                doc = pdfium.PdfDocument(str(pdf_path))
                page = doc[cand["page_index"]]
                try:
                    bitmap = page.render(scale=scale)
                    pil_img = bitmap.to_pil().convert("RGB")
                finally:
                    page.close()
            except Exception as exc:
                print(f"  [warn] Render failed {pdf_path.name} p{cand['page_num']}: {exc}")
                return False
            finally:
                if doc is not None:
                    doc.close()

            label_conf = cand["label_confidence"]
            if cand["page_type"] == "scanned":
                screen_res = screener.screen_pii(pil_img)
                if screen_res["has_pii"]:
                    ocr_pii_rejected += 1
                    print(f"  [PII-OCR rejected] {res_id} p{cand['page_num']}")
                    return False
                pii_status = "clean_ocr"
                if cand["page_role"] == "first_page_header" and label_conf == "needs_visual_verify":
                    if screen_res.get("detected_saraban_type") == st:
                        label_conf = "high"
                quality_tags = screen_res.get("quality_tags") or detect_quality_tags(pil_img)
            else:
                pii_status = "clean_text"
                quality_tags = detect_quality_tags(pil_img)

            out_dir = SARABAN_PAGES_DIR / st
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / f"{res_id}_p{cand['page_num']}.png"
            pil_img.save(str(img_path), format="PNG")
            width, height = pil_img.size

            used_per_pdf[res_id] += 1
            st_rows.append(
                {
                    "image": str(img_path),
                    "doc_id": rec["doc_id"],
                    "resource_id": res_id,
                    "saraban_type": st,
                    "saraban_subtype": cand["saraban_subtype"],
                    "page_role": cand["page_role"],
                    "page_type": cand["page_type"],
                    "label_confidence": label_conf,
                    "pii_screen_status": pii_status,
                    "has_pua_mojibake": bool(cand["has_pua_mojibake"]),
                    "image_coverage_ratio": float(cand["image_coverage_ratio"]),
                    "layout_complexity_score": float(cand["layout_complexity_score"]),
                    "path_count": int(cand["path_count"]),
                    "text_rect_count": int(cand["text_rect_count"]),
                    "has_embedded_text": cand["embedded_text_chars"] >= 20,
                    "embedded_text_chars": int(cand["embedded_text_chars"]),
                    "embedded_text_preview": cand["embedded_text_preview"],
                    "meta_doc_no": rec["meta_doc_no"],
                    "meta_doc_date": rec["meta_doc_date"],
                    "meta_subject": rec["meta_subject"],
                    "organization": rec["organization"],
                    "source_portal": rec["source_portal"],
                    "source_url": rec["source_url"],
                    "page_num": int(cand["page_num"]),
                    "total_pages": int(cand["total_pages"]),
                    "sha256": rec["sha256"],
                    "width": width,
                    "height": height,
                    "ground_truth_text": "",
                    "gt_status": "unverified",
                    "quality_tags": quality_tags,
                }
            )
            return True

        target_digital = per_type - round(per_type * scanned_ratio)
        if st in {"internal_memo", "order_regulation", "public_relations", "official_record"}:
            target_digital += 3
        target_scanned_cap = per_type - target_digital

        picked_scanned_count = 0
        for cand in scanned_pool:
            if picked_scanned_count >= target_scanned_cap:
                break
            if try_render_and_add(cand):
                picked_scanned_count += 1

        for cand in scanned_ocr_pool:
            if picked_scanned_count >= target_scanned_cap:
                break
            if try_render_and_add(cand):
                picked_scanned_count += 1

        picked_digital_count = 0
        for cand in digital_pool:
            if picked_digital_count >= target_digital:
                break
            if try_render_and_add(cand):
                picked_digital_count += 1

        if len(st_rows) < per_type:
            already_keys = {(r["resource_id"], r["page_num"]) for r in st_rows}
            for cand in scanned_pool + scanned_ocr_pool:
                if len(st_rows) >= per_type:
                    break
                if (cand["resource_id"], cand["page_num"]) in already_keys:
                    continue
                if try_render_and_add(cand):
                    picked_scanned_count += 1
                    already_keys.add((cand["resource_id"], cand["page_num"]))

        print(
            f"[{st}] built {len(st_rows)} pages "
            f"(scanned+ocr={picked_scanned_count}, digital={picked_digital_count})"
        )
        rows.extend(st_rows)

    if not rows:
        raise RuntimeError("No Saraban pages built.")

    features = Features(
        {
            "image": Image(),
            "doc_id": Value("string"),
            "resource_id": Value("string"),
            "saraban_type": Value("string"),
            "saraban_subtype": Value("string"),
            "page_role": Value("string"),
            "page_type": Value("string"),
            "label_confidence": Value("string"),
            "pii_screen_status": Value("string"),
            "has_pua_mojibake": Value("bool"),
            "image_coverage_ratio": Value("float32"),
            "layout_complexity_score": Value("float32"),
            "path_count": Value("int32"),
            "text_rect_count": Value("int32"),
            "has_embedded_text": Value("bool"),
            "embedded_text_chars": Value("int32"),
            "embedded_text_preview": Value("string"),
            "meta_doc_no": Value("string"),
            "meta_doc_date": Value("string"),
            "meta_subject": Value("string"),
            "organization": Value("string"),
            "source_portal": Value("string"),
            "source_url": Value("string"),
            "page_num": Value("int32"),
            "total_pages": Value("int32"),
            "sha256": Value("string"),
            "width": Value("int32"),
            "height": Value("int32"),
            "ground_truth_text": Value("string"),
            "gt_status": Value("string"),
            "quality_tags": Sequence(Value("string")),
        }
    )

    ds = Dataset.from_list(rows, features=features)
    ds_dict = DatasetDict({"test": ds})
    ds_dict.save_to_disk(str(SARABAN_HF_DIR))

    type_counts = dict(Counter(r["page_type"] for r in rows))
    scanned_total = type_counts.get("scanned", 0) + type_counts.get("scanned_ocr", 0)
    stats = {
        "total_pages": len(rows),
        "total_documents": len({r["resource_id"] for r in rows}),
        "total_organizations": len({r["organization"] for r in rows}),
        "by_saraban_type": dict(Counter(r["saraban_type"] for r in rows)),
        "by_saraban_subtype": dict(Counter(r["saraban_subtype"] for r in rows)),
        "by_page_type": type_counts,
        "by_page_role": dict(Counter(r["page_role"] for r in rows)),
        "by_source_portal": dict(Counter(r["source_portal"] for r in rows)),
        "scanned_share_pct": round(100.0 * scanned_total / len(rows), 2),
        "digital_share_pct": round(100.0 * type_counts.get("digital", 0) / len(rows), 2),
        "ocr_pii_rejected": ocr_pii_rejected,
        "dpi": dpi,
    }
    (SARABAN_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nBuilt Saraban HF Dataset: {stats['total_pages']} pages from {stats['total_documents']} PDFs")
    print(f"By saraban_type: {stats['by_saraban_type']}")
    print(
        f"By page_type: {stats['by_page_type']} "
        f"(scanned={stats['scanned_share_pct']}%, digital={stats['digital_share_pct']}%)"
    )
    print(f"OCR PII rejected: {ocr_pii_rejected}")


def main_general(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build stratified OCR benchmark dataset.")
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL_PAGES)
    parser.add_argument("--scanned-ratio", type=float, default=SCANNED_RATIO)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_PER_PDF)
    parser.add_argument("--dpi", type=int, default=RENDER_DPI)
    args = parser.parse_args(argv)
    build_general_dataset(
        target_total=args.target_total,
        scanned_ratio=args.scanned_ratio,
        max_pages_per_pdf=args.max_pages,
        dpi=args.dpi,
    )


def main_saraban(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build 300 DPI stratified Saraban 6-type OCR benchmark.")
    parser.add_argument("--per-type", type=int, default=SARABAN_TARGET_PER_TYPE)
    parser.add_argument("--scanned-ratio", type=float, default=SARABAN_SCANNED_RATIO)
    parser.add_argument("--max-pages", type=int, default=SARABAN_MAX_PAGES_PER_PDF)
    parser.add_argument("--dpi", type=int, default=SARABAN_RENDER_DPI)
    parser.add_argument("--screener-url", type=str, default=None)
    args = parser.parse_args(argv)
    build_saraban_dataset(
        per_type=args.per_type,
        scanned_ratio=args.scanned_ratio,
        max_pages_per_pdf=args.max_pages,
        dpi=args.dpi,
        screener_url=args.screener_url,
    )


def infer_saraban_type_from_keyword(keyword: str) -> str:
    """Infer coarse Thai Saraban document category from a Google search keyword."""
    kw = keyword or ""
    if "บันทึกข้อความ" in kw:
        return "internal_memo"
    if any(k in kw for k in ("คำสั่ง", "ระเบียบ", "ข้อบังคับ")):
        return "order_regulation"
    if any(k in kw for k in ("ประกาศ", "จัดซื้อจัดจ้าง")):
        return "public_relations"
    if "รายงานการประชุม" in kw:
        return "official_record"
    if any(k in kw for k in ("หนังสือราชการ", "เอกสารราชการ")):
        return "external_letter"
    return "other"


def preprocess_document_image(pil_img: PILImage.Image, max_edge: int = 3508) -> PILImage.Image:
    """Normalize EXIF orientation, convert to RGB, and cap longest edge if max_edge > 0."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    if max_edge > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_edge:
            scale = max_edge / longest
            img = img.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                PILImage.Resampling.LANCZOS,
            )
    return img


def build_google_dataset(
    filtered_dir: Path = GOOGLE_CRAWL_DIR / "filtered",
    preprocessed_dir: Path = GOOGLE_PREPROCESSED_DIR,
    hf_dir: Path = GOOGLE_HF_DIR,
    max_edge: int = 3508,
    screener_url: str | None = None,
) -> dict[str, Any]:
    """Preprocess kept Google Images (EXIF fix, RGB PNG, PII check, quality tags) and save HF Dataset."""
    import hashlib
    import shutil
    from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value

    kept_dir = filtered_dir / "kept"
    report_path = filtered_dir / "report.json"
    if not kept_dir.exists():
        raise FileNotFoundError(f"Filtered kept directory not found: {kept_dir}")

    report_by_rel: dict[str, dict] = {}
    if report_path.exists():
        for rec in json.loads(report_path.read_text(encoding="utf-8")):
            if rec.get("relative_path"):
                report_by_rel[rec["relative_path"]] = rec

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    kept_files = sorted(
        p for p in kept_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts
    )
    if not kept_files:
        raise RuntimeError(f"No kept images found under {kept_dir}")

    if preprocessed_dir.exists():
        shutil.rmtree(preprocessed_dir)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    screener = ScreenerClient(base_url=screener_url) if screener_url else None
    rows: list[dict[str, Any]] = []
    pii_rejected = 0

    for src_path in kept_files:
        rel = src_path.relative_to(kept_dir)
        rel_posix = str(rel.as_posix())
        keyword = src_path.parent.name if src_path.parent != kept_dir else ""
        meta = report_by_rel.get(rel_posix, {})

        with PILImage.open(src_path) as raw_img:
            pil_img = preprocess_document_image(raw_img, max_edge=max_edge)

        has_pii = bool(meta.get("has_pii", False))
        if not has_pii and screener is not None:
            has_pii = bool(screener.screen_pii(pil_img).get("has_pii", False))

        if has_pii:
            pii_rejected += 1
            print(f"  [PII rejected] {rel_posix}")
            continue

        out_path = (preprocessed_dir / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(str(out_path), format="PNG", optimize=True)
        width, height = pil_img.size
        sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

        rows.append(
            {
                "image": str(out_path),
                "doc_id": sha256[:16],
                "keyword": keyword,
                "saraban_type_hint": infer_saraban_type_from_keyword(keyword),
                "relative_path": rel_posix,
                "source_portal": "google_images",
                "pii_screen_status": "clean_ocr",
                "ocr_boxes": int(meta.get("ocr_boxes", 0)),
                "ocr_chars": int(meta.get("ocr_chars", 0)),
                "mean_saturation": float(meta.get("mean_saturation", 0.0)),
                "light_ratio": float(meta.get("light_ratio", 0.0)),
                "width": width,
                "height": height,
                "sha256": sha256,
                "ground_truth_text": "",
                "gt_status": "unverified",
                "quality_tags": detect_quality_tags(pil_img),
            }
        )

    if not rows:
        raise RuntimeError("No clean images left after preprocessing/PII check.")

    features = Features(
        {
            "image": Image(),
            "doc_id": Value("string"),
            "keyword": Value("string"),
            "saraban_type_hint": Value("string"),
            "relative_path": Value("string"),
            "source_portal": Value("string"),
            "pii_screen_status": Value("string"),
            "ocr_boxes": Value("int32"),
            "ocr_chars": Value("int32"),
            "mean_saturation": Value("float32"),
            "light_ratio": Value("float32"),
            "width": Value("int32"),
            "height": Value("int32"),
            "sha256": Value("string"),
            "ground_truth_text": Value("string"),
            "gt_status": Value("string"),
            "quality_tags": Sequence(Value("string")),
        }
    )

    ds = Dataset.from_list(rows, features=features)
    ds_dict = DatasetDict({"test": ds})
    ds_dict.save_to_disk(str(hf_dir))

    stats = {
        "total_pages": len(rows),
        "total_documents": len(rows),
        "by_keyword": dict(Counter(r["keyword"] for r in rows)),
        "by_saraban_type_hint": dict(Counter(r["saraban_type_hint"] for r in rows)),
        "pii_rejected": pii_rejected,
        "max_edge": max_edge,
    }
    stats_path = hf_dir.parent / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built Google Docs HF Dataset: {stats['total_pages']} pages -> {hf_dir}")
    print(f"By keyword: {stats['by_keyword']}")
    print(f"By saraban_type_hint: {stats['by_saraban_type_hint']} (PII rejected={pii_rejected})")
    return stats


def main_google(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess filtered Google Images (EXIF fix, RGB PNG, quality tags) and build HF dataset."
    )
    parser.add_argument("--filtered-dir", type=Path, default=GOOGLE_CRAWL_DIR / "filtered")
    parser.add_argument("--preprocessed-dir", type=Path, default=GOOGLE_PREPROCESSED_DIR)
    parser.add_argument("--hf-dir", type=Path, default=GOOGLE_HF_DIR)
    parser.add_argument("--max-edge", type=int, default=3508)
    parser.add_argument("--screener-url", type=str, default=None)
    args = parser.parse_args(argv)
    build_google_dataset(
        filtered_dir=args.filtered_dir,
        preprocessed_dir=args.preprocessed_dir,
        hf_dir=args.hf_dir,
        max_edge=args.max_edge,
        screener_url=args.screener_url,
    )
