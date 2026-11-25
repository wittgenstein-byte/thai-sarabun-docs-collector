"""Draft Ground Truth for Config 4 (dla_bulk, 120-page stratified subset) using KKU GenAI VLMs."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from config import KKU_BASE_URL, KKU_MODEL_A, KKU_MODEL_B, load_gt_api_keys, load_kku_api_key
from draft_google_gt import call_iapp_openthai
from draft_gt_dual import call_vlm_openai_compat, compute_agreement
from train_ocr.common.settings import DLA_BULK_DIR, DLA_BULK_HF_DIR, DLA_BULK_PAGES_DIR

DLA_BULK_GT_DIR = DLA_BULK_DIR / "gt"
DLA_BULK_GT_MANIFEST_PATH = DLA_BULK_GT_DIR / "dla_bulk_manifest.jsonl"
DLA_BULK_GT_DRAFTS_PATH = DLA_BULK_GT_DIR / "dual_drafts.jsonl"
DLA_BULK_GT_ANNOTATIONS_PATH = DLA_BULK_GT_DIR / "gt_annotations.jsonl"


def build_dla_bulk_manifest(
    hf_dir: Path = DLA_BULK_HF_DIR,
    pages_dir: Path = DLA_BULK_PAGES_DIR,
    manifest_path: Path = DLA_BULK_GT_MANIFEST_PATH,
    target: int = 120,
) -> list[dict]:
    """Generate stratified 120-page manifest from dla_bulk HF dataset."""
    from datasets import DatasetDict

    if not hf_dir.exists():
        raise FileNotFoundError(f"DLA bulk HF dataset not found: {hf_dir}")

    ds = DatasetDict.load_from_disk(str(hf_dir))["train"]
    all_records: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        sid = f"{row['doc_id']}_p{row['page_num']}"
        img_path = pages_dir / row["source"] / f"{sid}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Page image missing: {img_path}")
        all_records.append(
            {
                "sample_id": sid,
                "doc_id": row["doc_id"],
                "image_path": str(img_path.as_posix()),
                "source": row["source"],
                "saraban_type": row["source"],
                "page_type": row["page_type"],
                "page_role": row.get("subkey") or row["source"],
                "organization": row.get("dept") or row.get("title") or row["source"],
                "meta_doc_no": row.get("doc_no") or "",
                "meta_doc_date": row.get("doc_date") or "",
                "meta_subject": row.get("title") or "",
                "embedded_text_preview": row.get("embedded_text_preview") or "",
                "page_num": int(row["page_num"]),
                "total_pages": int(row["total_pages"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "quality_tags": [],
            }
        )

    if target > 0 and target < len(all_records):
        by_source: dict[str, list[dict]] = defaultdict(list)
        for rec in all_records:
            by_source[rec["source"]].append(rec)
        # Prioritize diverse page_type and unique doc_id within each source
        for src in by_source:
            by_source[src].sort(key=lambda r: (r["page_num"], r["page_type"], r["sample_id"]))
        selected: list[dict] = []
        sources = sorted(by_source)
        while len(selected) < target and any(by_source[s] for s in sources):
            for src in sources:
                if by_source[src] and len(selected) < target:
                    selected.append(by_source[src].pop(0))
        records = selected
    else:
        records = all_records

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def run_dla_bulk_drafting(
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
    """Build manifest and generate dual-model KKU VLM drafts for dla_bulk."""
    api_key = load_kku_api_key()
    manifest = build_dla_bulk_manifest(target=target)
    if limit > 0:
        manifest = manifest[:limit]

    existing: dict[str, dict] = {}
    if DLA_BULK_GT_DRAFTS_PATH.exists():
        for line in DLA_BULK_GT_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
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
        f"Pending dla_bulk samples: {len(todo)}/{len(manifest)} "
        f"(workers={workers}, models={model_a} + {model_b}, fallback={fallback_model}, only={only})",
        flush=True,
    )
    if not todo:
        return

    lock = threading.Lock()
    processed = 0

    def process_one(item: dict) -> None:
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
                DLA_BULK_GT_DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with DLA_BULK_GT_DRAFTS_PATH.open("w", encoding="utf-8") as f:
                    for r in existing.values():
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                processed += 1
                print(
                    f"[{processed}/{len(todo)}] {sid} ({item['source']}/{item['page_type']}) "
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
        f"\nDone: {len(existing)}/{len(manifest)} dla_bulk drafts stored in {DLA_BULK_GT_DRAFTS_PATH}",
        flush=True,
    )


def run_redraft_unverified_iapp(workers: int = 4, delay: float = 0.1) -> None:
    """Re-draft only unverified dla_bulk pages with iApp OpenThai OCR 2.1 (replacing hallucinated drafts)."""
    _, iapp_key = load_gt_api_keys()
    if not iapp_key:
        raise ValueError("IAPP_API_KEY / GT_THAI_DOCS not found in .env")

    manifest = [
        json.loads(l)
        for l in DLA_BULK_GT_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    reviewed: set[str] = set()
    if DLA_BULK_GT_ANNOTATIONS_PATH.exists():
        for l in DLA_BULK_GT_ANNOTATIONS_PATH.read_text(encoding="utf-8").splitlines():
            if l.strip():
                rec = json.loads(l)
                if rec.get("gt_status") in {"human_verified", "flagged"}:
                    reviewed.add(rec["sample_id"])

    existing: dict[str, dict] = {}
    if DLA_BULK_GT_DRAFTS_PATH.exists():
        for l in DLA_BULK_GT_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
            if l.strip():
                rec = json.loads(l)
                existing[rec["sample_id"]] = rec

    todo = [item for item in manifest if item["sample_id"] not in reviewed]
    print(f"Re-drafting {len(todo)} unverified dla_bulk pages with iApp OpenThai OCR 2.1 (workers={workers})...", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    processed = 0

    def process_one(item: dict) -> None:
        nonlocal processed
        sid = item["sample_id"]
        img_path = Path(item["image_path"])
        session = requests.Session()
        try:
            md = call_iapp_openthai(session, iapp_key, img_path)
        except Exception as exc:
            print(f"  [err] iApp failed for {sid}: {exc}", flush=True)
            return

        with lock:
            rec = dict(
                existing.get(
                    sid,
                    {
                        "sample_id": sid,
                        "model_a_name": "iapp-openthai-2.1",
                        "model_a_markdown": "",
                        "model_a_kie": {},
                        "model_b_name": "",
                        "model_b_markdown": "",
                        "model_b_kie": {},
                        "agreement_score": 0.0,
                    },
                )
            )
            rec["model_a_name"] = "iapp-openthai-2.1"
            rec["model_a_markdown"] = md
            rec["model_a_kie"] = {}
            rec["agreement_score"] = compute_agreement(md, rec.get("model_b_markdown", ""))
            existing[sid] = rec
            with DLA_BULK_GT_DRAFTS_PATH.open("w", encoding="utf-8") as f:
                for r in existing.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            processed += 1
            print(f"[{processed}/{len(todo)}] {sid} ({item['source']}/{item['page_type']}) chars={len(md)}", flush=True)
        if delay > 0:
            time.sleep(delay)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, item) for item in todo]
        for fut in as_completed(futures):
            fut.result()

    print(f"\nDone re-drafting {processed}/{len(todo)} unverified pages with iApp OpenThai OCR 2.1.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft GT for dla_bulk via KKU GenAI VLMs.")
    parser.add_argument("--base-url", type=str, default=KKU_BASE_URL)
    parser.add_argument("--model-a", type=str, default=KKU_MODEL_A)
    parser.add_argument("--model-b", type=str, default=KKU_MODEL_B)
    parser.add_argument("--fallback-model", type=str, default="qwen3.8-27b")
    parser.add_argument("--only", choices=["both", "a", "b"], default="both")
    parser.add_argument("--target", type=int, default=120, help="Stratified subset size (0 = all 1649).")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N pages (0 = all).")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--redraft-unverified-iapp", action="store_true", help="Re-draft unverified pages using iApp OpenThai OCR 2.1.")
    args = parser.parse_args()
    if args.redraft_unverified_iapp:
        run_redraft_unverified_iapp(workers=args.workers, delay=args.delay)
        return
    run_dla_bulk_drafting(
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
