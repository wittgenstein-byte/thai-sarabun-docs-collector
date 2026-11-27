"""Draft Ground Truth for Config 7 (lawgoth, 120-page stratified subset) using KKU GenAI VLMs."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from config import KKU_BASE_URL, KKU_MODEL_A, KKU_MODEL_B, load_kku_api_key
from draft_gt_dual import call_vlm_openai_compat, compute_agreement
from train_ocr.common.settings import LAWGOTH_DIR, LAWGOTH_HF_DIR, LAWGOTH_PAGES_DIR

LAWGOTH_GT_DIR = LAWGOTH_DIR / "gt"
LAWGOTH_GT_MANIFEST_PATH = LAWGOTH_GT_DIR / "lawgoth_manifest.jsonl"
LAWGOTH_GT_DRAFTS_PATH = LAWGOTH_GT_DIR / "dual_drafts.jsonl"
LAWGOTH_GT_ANNOTATIONS_PATH = LAWGOTH_GT_DIR / "gt_annotations.jsonl"


def _pick_unique_docs(candidates: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    """Select up to `quota` items prioritizing unique doc_id and diverse file_type/page_num."""
    picked: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    # First pass: unique doc_id
    for rec in candidates:
        if len(picked) >= quota:
            break
        if rec["doc_id"] not in seen_docs:
            seen_docs.add(rec["doc_id"])
            picked.append(rec)
    # Second pass if still under quota
    if len(picked) < quota:
        picked_ids = {r["sample_id"] for r in picked}
        for rec in candidates:
            if len(picked) >= quota:
                break
            if rec["sample_id"] not in picked_ids:
                picked.append(rec)
    return picked


def build_lawgoth_manifest(
    hf_dir: Path = LAWGOTH_HF_DIR,
    pages_dir: Path = LAWGOTH_PAGES_DIR,
    manifest_path: Path = LAWGOTH_GT_MANIFEST_PATH,
    target: int = 120,
) -> list[dict[str, Any]]:
    """Generate stratified 120-page manifest from lawgoth HF dataset."""
    from datasets import DatasetDict

    if not hf_dir.exists():
        raise FileNotFoundError(f"LawGoTh HF dataset not found: {hf_dir}")

    ds = DatasetDict.load_from_disk(str(hf_dir))["train"]
    all_records: list[dict[str, Any]] = []
    for i in range(len(ds)):
        row = ds[i]
        sid = f"{row['doc_id']}_p{row['page_num']}"
        img_path = pages_dir / f"{sid}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Page image missing: {img_path}")
        has_pua = bool(row.get("has_pua_mojibake", False))
        file_type = row.get("file_type") or "content"
        page_type = row.get("page_type") or "digital"
        all_records.append(
            {
                "sample_id": sid,
                "doc_id": row["doc_id"],
                "image_path": str(img_path.as_posix()),
                "file_type": file_type,
                "saraban_type": file_type,
                "page_type": page_type,
                "page_role": "pua_mojibake" if has_pua else page_type,
                "organization": row.get("law_name_th") or "law.go.th",
                "meta_doc_no": str(row.get("law_id") or ""),
                "meta_doc_date": (row.get("annouce_date") or "")[:10],
                "meta_subject": row.get("law_name_th") or "",
                "embedded_text_preview": "" if has_pua else (row.get("embedded_text_preview") or ""),
                "has_pua_mojibake": has_pua,
                "page_num": int(row["page_num"]),
                "total_pages": int(row["total_pages"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "quality_tags": [],
            }
        )

    if target > 0 and target < len(all_records):
        # Prioritize Thai legal documents (exclude files_en unless needed)
        thai_records = [r for r in all_records if r["file_type"] != "files_en"]
        pool = thai_records if len(thai_records) >= target else all_records

        # Sort each bucket to mix relate_external/agency/dga + content and page_num
        def _sort_key(r: dict[str, Any]) -> tuple[int, int, str]:
            ft_prio = 0 if r["file_type"] in {"relate_external", "agency", "dga"} else 1
            return (ft_prio, r["page_num"] % 3, r["sample_id"])

        scanned_pool = sorted([r for r in pool if r["page_type"] in {"scanned", "scanned_ocr"}], key=_sort_key)
        pua_pool = sorted([r for r in pool if r["page_type"] == "digital" and r["has_pua_mojibake"]], key=_sort_key)
        clean_pool = sorted([r for r in pool if r["page_type"] == "digital" and not r["has_pua_mojibake"]], key=_sort_key)

        q_scanned = round(target * 0.50)
        q_pua = round(target * 0.25)
        q_clean = target - q_scanned - q_pua

        selected = (
            _pick_unique_docs(scanned_pool, q_scanned)
            + _pick_unique_docs(pua_pool, q_pua)
            + _pick_unique_docs(clean_pool, q_clean)
        )
        records = selected[:target]
    else:
        records = all_records

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def run_lawgoth_drafting(
    base_url: str = KKU_BASE_URL,
    model_a: str = KKU_MODEL_A,
    model_b: str = KKU_MODEL_B,
    fallback_model: str = "qwen3.8-27b",
    only: str = "both",
    target: int = 120,
    limit: int = 0,
    delay: float = 0.1,
    workers: int = 6,
) -> None:
    """Build manifest and generate dual-model KKU VLM drafts for lawgoth."""
    api_key = load_kku_api_key()
    manifest = build_lawgoth_manifest(target=target)
    if limit > 0:
        manifest = manifest[:limit]

    existing: dict[str, dict[str, Any]] = {}
    if LAWGOTH_GT_DRAFTS_PATH.exists():
        for line in LAWGOTH_GT_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["sample_id"]] = rec

    todo = [
        item
        for item in manifest
        if (only in {"both", "a"} and not existing.get(item["sample_id"], {}).get("model_a_markdown"))
        or (only in {"both", "b"} and not existing.get(item["sample_id"], {}).get("model_b_markdown"))
    ]
    print(
        f"Pending lawgoth samples: {len(todo)}/{len(manifest)} "
        f"(workers={workers}, models={model_a} + {model_b}, fallback={fallback_model}, only={only})",
        flush=True,
    )
    if not todo:
        return

    lock = threading.Lock()
    processed = 0

    def process_one(item: dict[str, Any]) -> None:
        nonlocal processed
        sid = item["sample_id"]
        img_path = Path(item["image_path"])

        with lock:
            rec = dict(
                existing.get(
                    sid,
                    {
                        "sample_id": sid,
                        "model_a_name": model_a,
                        "model_a_markdown": "",
                        "model_a_kie": {},
                        "model_b_name": model_b,
                        "model_b_markdown": "",
                        "model_b_kie": {},
                        "agreement_score": 0.0,
                    },
                )
            )

        session = requests.Session()
        changed = False

        if only in {"both", "a"} and not rec.get("model_a_markdown"):
            try:
                md, kie = call_vlm_openai_compat(session, base_url, api_key, model_a, img_path)
                rec["model_a_name"] = model_a
                rec["model_a_markdown"] = md
                rec["model_a_kie"] = kie
                changed = True
            except Exception as exc:
                print(f"  [warn] {model_a} failed for {sid} ({exc}); trying {fallback_model}", flush=True)
                try:
                    md, kie = call_vlm_openai_compat(session, base_url, api_key, fallback_model, img_path)
                    rec["model_a_name"] = fallback_model
                    rec["model_a_markdown"] = md
                    rec["model_a_kie"] = kie
                    changed = True
                except Exception as exc2:
                    print(f"  [err] {fallback_model} failed for {sid}: {exc2}", flush=True)

        if only in {"both", "b"} and not rec.get("model_b_markdown"):
            try:
                md, kie = call_vlm_openai_compat(session, base_url, api_key, model_b, img_path)
                rec["model_b_name"] = model_b
                rec["model_b_markdown"] = md
                rec["model_b_kie"] = kie
                changed = True
            except Exception as exc:
                print(f"  [err] {model_b} failed for {sid}: {exc}", flush=True)

        if changed:
            rec["agreement_score"] = compute_agreement(
                rec.get("model_a_markdown", ""), rec.get("model_b_markdown", "")
            )
            with lock:
                existing[sid] = rec
                LAWGOTH_GT_DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with LAWGOTH_GT_DRAFTS_PATH.open("w", encoding="utf-8") as f:
                    for r in existing.values():
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                processed += 1
                print(
                    f"[{processed}/{len(todo)}] {sid} ({item['file_type']}/{item['page_role']}) "
                    f"agreement={rec['agreement_score']:.3f}",
                    flush=True,
                )
            if delay > 0:
                time.sleep(delay)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, item) for item in todo]
        for fut in as_completed(futures):
            fut.result()

    print(
        f"\nDone: {len(existing)}/{len(manifest)} lawgoth drafts stored in {LAWGOTH_GT_DRAFTS_PATH}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft GT for Config 7 (lawgoth, 120-page subset) via KKU VLMs.")
    parser.add_argument("--base-url", default=KKU_BASE_URL)
    parser.add_argument("--model-a", default=KKU_MODEL_A)
    parser.add_argument("--model-b", default=KKU_MODEL_B)
    parser.add_argument("--fallback-model", default="qwen3.8-27b")
    parser.add_argument("--only", choices=["both", "a", "b"], default="both")
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run_lawgoth_drafting(
        base_url=args.base_url,
        model_a=args.model_a,
        model_b=args.model_b,
        fallback_model=args.fallback_model,
        only=args.only,
        target=args.target,
        limit=args.limit,
        delay=args.delay,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
