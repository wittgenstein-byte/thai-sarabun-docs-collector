"""Select a stratified 120-page Golden Set (20 x 6 Saraban types, 75% scanned / 25% digital)."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import DatasetDict

from config import (
    GOLDEN_DIR,
    GOLDEN_MANIFEST_PATH,
    SARABAN_HF_DIR,
    SARABAN_PAGES_DIR,
    SARABAN_TYPES,
)


def pick_with_diversity(
    candidates: list[dict],
    target: int,
    used_pdfs: dict[str, int],
    already_ids: set[str],
    max_per_pdf_pass: tuple[int, ...] = (1, 2),
) -> list[dict]:
    """Pick up to `target` candidates preferring unique PDFs and diverse page_roles."""
    role_order = {"first_page_header": 0, "signature_page": 1, "body_table": 2}
    # Sort so first_page_header -> signature_page -> body_table, with higher complexity first for digital
    ordered = sorted(
        candidates,
        key=lambda x: (
            role_order.get(x["page_role"], 9),
            -float(x.get("layout_complexity_score", 0.0)),
            x["sample_id"],
        ),
    )

    picked: list[dict] = []
    for cap in max_per_pdf_pass:
        # Interleave roles in each pass so we get ~60% header, ~25% signature, ~15% body_table
        by_role: dict[str, list[dict]] = defaultdict(list)
        for item in ordered:
            if item["sample_id"] in already_ids:
                continue
            by_role[item["page_role"]].append(item)

        role_cycle = [
            "first_page_header",
            "first_page_header",
            "signature_page",
            "first_page_header",
            "body_table",
        ]
        idx_map = {r: 0 for r in ("first_page_header", "signature_page", "body_table")}
        progressed = True
        while len(picked) < target and progressed:
            progressed = False
            for r in role_cycle:
                if len(picked) >= target:
                    break
                items = by_role.get(r, [])
                while idx_map[r] < len(items):
                    cand = items[idx_map[r]]
                    idx_map[r] += 1
                    sid = cand["sample_id"]
                    rid = cand["resource_id"]
                    if sid in already_ids or used_pdfs[rid] >= cap:
                        continue
                    already_ids.add(sid)
                    used_pdfs[rid] += 1
                    picked.append(cand)
                    progressed = True
                    break
            # If role_cycle exhausted, drain any remaining role
            if not progressed and len(picked) < target:
                for r in ("first_page_header", "signature_page", "body_table"):
                    items = by_role.get(r, [])
                    while idx_map[r] < len(items) and len(picked) < target:
                        cand = items[idx_map[r]]
                        idx_map[r] += 1
                        sid = cand["sample_id"]
                        rid = cand["resource_id"]
                        if sid in already_ids or used_pdfs[rid] >= cap:
                            continue
                        already_ids.add(sid)
                        used_pdfs[rid] += 1
                        picked.append(cand)
                        progressed = True
                        break
    return picked


def select_golden(per_type: int = 20, scanned_ratio: float = 0.75) -> None:
    if not SARABAN_HF_DIR.exists():
        raise FileNotFoundError(f"Saraban HF dataset not found at {SARABAN_HF_DIR}")

    ds = DatasetDict.load_from_disk(str(SARABAN_HF_DIR))["test"]
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    fallback_img_dir = GOLDEN_DIR / "images"

    all_rows: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        sid = f"{row['resource_id']}_p{row['page_num']}"
        expected_png = SARABAN_PAGES_DIR / row["saraban_type"] / f"{sid}.png"
        if expected_png.exists():
            img_path = str(expected_png.as_posix())
        else:
            fallback_img_dir.mkdir(parents=True, exist_ok=True)
            out_png = fallback_img_dir / f"{sid}.png"
            if not out_png.exists():
                row["image"].save(str(out_png), format="PNG")
            img_path = str(out_png.as_posix())

        all_rows.append(
            {
                "sample_id": sid,
                "dataset_index": i,
                "image_path": img_path,
                "doc_id": row["doc_id"],
                "resource_id": row["resource_id"],
                "saraban_type": row["saraban_type"],
                "saraban_subtype": row["saraban_subtype"],
                "page_role": row["page_role"],
                "page_type": row["page_type"],
                "label_confidence": row["label_confidence"],
                "has_pua_mojibake": bool(row["has_pua_mojibake"]),
                "layout_complexity_score": float(row["layout_complexity_score"]),
                "embedded_text_chars": int(row["embedded_text_chars"]),
                "embedded_text_preview": row["embedded_text_preview"],
                "meta_doc_no": row["meta_doc_no"],
                "meta_doc_date": row["meta_doc_date"],
                "meta_subject": row["meta_subject"],
                "organization": row["organization"],
                "source_portal": row["source_portal"],
                "source_url": row["source_url"],
                "page_num": int(row["page_num"]),
                "total_pages": int(row["total_pages"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "quality_tags": list(row["quality_tags"]),
            }
        )

    selected: list[dict] = []
    for st in SARABAN_TYPES:
        pool = [r for r in all_rows if r["saraban_type"] == st]
        scanned_pool = [r for r in pool if r["page_type"] in {"scanned", "scanned_ocr"}]
        digital_pool = [r for r in pool if r["page_type"] == "digital"]

        # Bump digital target slightly on digital-rich types to offset stamped_letter (only 1 digital)
        target_digital = per_type - round(per_type * scanned_ratio)
        if st in {"internal_memo", "order_regulation", "public_relations", "official_record"}:
            target_digital += 1
        target_scanned = per_type - target_digital

        used_pdfs: dict[str, int] = defaultdict(int)
        already_ids: set[str] = set()

        st_picked: list[dict] = []
        st_picked.extend(pick_with_diversity(scanned_pool, target_scanned, used_pdfs, already_ids))
        st_picked.extend(pick_with_diversity(digital_pool, target_digital, used_pdfs, already_ids))

        if len(st_picked) < per_type:
            rem = per_type - len(st_picked)
            st_picked.extend(pick_with_diversity(scanned_pool + digital_pool, rem, used_pdfs, already_ids))

        selected.extend(st_picked)
        print(
            f"[{st:17s}] {len(st_picked):2d} pages | "
            f"types={dict(Counter(x['page_type'] for x in st_picked))} | "
            f"roles={dict(Counter(x['page_role'] for x in st_picked))}"
        )

    with GOLDEN_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    pt_counts = Counter(x["page_type"] for x in selected)
    sc_tot = pt_counts.get("scanned", 0) + pt_counts.get("scanned_ocr", 0)
    print(f"\nSaved Golden Set: {len(selected)} pages -> {GOLDEN_MANIFEST_PATH}")
    print(f"Unique PDFs: {len({x['resource_id'] for x in selected})} | Scanned: {100*sc_tot/len(selected):.1f}% | Digital: {100*pt_counts.get('digital',0)/len(selected):.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select stratified 120-page Golden Set.")
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--scanned-ratio", type=float, default=0.75)
    args = parser.parse_args()
    select_golden(per_type=args.per_type, scanned_ratio=args.scanned_ratio)


if __name__ == "__main__":
    main()
