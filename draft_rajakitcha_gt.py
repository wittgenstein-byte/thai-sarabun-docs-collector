"""Draft Ground Truth for Config 6 (rajakitcha) using iApp OpenThai OCR 2.1 API (+ optional KKU VLM)."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from config import (
    KKU_BASE_URL,
    KKU_MODEL_A,
    load_gt_api_keys,
    load_kku_api_key,
)
from draft_google_gt import call_iapp_openthai
from draft_gt_dual import call_vlm_openai_compat, compute_agreement
from train_ocr.common.settings import (
    RAJAKITCHA_DIR,
    RAJAKITCHA_HF_DIR,
    RAJAKITCHA_PREPROCESSED_DIR,
)

RAJAKITCHA_GT_DIR = RAJAKITCHA_DIR / "gt"
RAJAKITCHA_GT_MANIFEST_PATH = RAJAKITCHA_GT_DIR / "rajakitcha_manifest.jsonl"
RAJAKITCHA_GT_DRAFTS_PATH = RAJAKITCHA_GT_DIR / "dual_drafts.jsonl"
RAJAKITCHA_GT_ANNOTATIONS_PATH = RAJAKITCHA_GT_DIR / "gt_annotations.jsonl"


def build_rajakitcha_manifest(
    hf_dir: Path = RAJAKITCHA_HF_DIR,
    preprocessed_dir: Path = RAJAKITCHA_PREPROCESSED_DIR,
    manifest_path: Path = RAJAKITCHA_GT_MANIFEST_PATH,
) -> list[dict[str, Any]]:
    """Generate rajakitcha_manifest.jsonl from data/rajakitcha/hf_dataset and preprocessed PNGs."""
    from datasets import DatasetDict

    if not hf_dir.exists():
        raise FileNotFoundError(f"Rajakitcha HF dataset not found: {hf_dir}")

    ds = DatasetDict.load_from_disk(str(hf_dir))["train"]
    total = len(ds)
    records: list[dict[str, Any]] = []
    for i in range(total):
        row = ds[i]
        rel = Path(row["filename"]).with_suffix(".png")
        img_path = preprocessed_dir / rel
        if not img_path.exists():
            raise FileNotFoundError(f"Preprocessed image missing: {img_path}")
        records.append(
            {
                "sample_id": row["doc_id"],
                "doc_id": row["doc_id"],
                "image_path": str(img_path.as_posix()),
                "relative_path": row["filename"],
                "category": row.get("category", "ราชกิจจา"),
                "subcategory": row.get("subcategory", "constituencies"),
                "saraban_type": "ราชกิจจา",
                "page_type": "scanned",
                "page_role": "rajakitcha",
                "organization": "ราชกิจจานุเบกษา",
                "page_num": i + 1,
                "total_pages": total,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "quality_tags": list(row.get("quality_tags") or []),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def run_rajakitcha_drafting(
    with_gemini: bool = False,
    limit: int = 0,
    delay: float = 0.2,
    workers: int = 4,
) -> None:
    """Build manifest (if needed) and draft GT for rajakitcha via iApp OpenThai OCR 2.1 API."""
    _, iapp_key = load_gt_api_keys()
    if not iapp_key:
        raise ValueError(
            "IAPP_API_KEY (or OPENTHAI_API_KEY) not found in environment or .env. "
            "Please add IAPP_API_KEY=... to .env before running."
        )

    kku_key = load_kku_api_key() if with_gemini else ""

    manifest = build_rajakitcha_manifest()
    if limit > 0:
        manifest = manifest[:limit]

    existing: dict[str, dict[str, Any]] = {}
    if RAJAKITCHA_GT_DRAFTS_PATH.exists():
        for line in RAJAKITCHA_GT_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["sample_id"]] = rec

    todo = [
        item
        for item in manifest
        if not existing.get(item["sample_id"], {}).get("model_b_markdown")
        or (with_gemini and not existing.get(item["sample_id"], {}).get("model_a_markdown"))
    ]
    print(
        f"Pending rajakitcha samples: {len(todo)}/{len(manifest)} "
        f"(iApp OpenThai 2.1{' + ' + KKU_MODEL_A if with_gemini else ''}, workers={workers})",
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
                        "model_a_name": KKU_MODEL_A if with_gemini else "",
                        "model_a_markdown": "",
                        "model_a_kie": {},
                        "model_b_name": "iapp-openthai-2.1",
                        "model_b_markdown": "",
                        "model_b_kie": {},
                        "agreement_score": 0.0,
                    },
                )
            )

        session = requests.Session()
        changed = False

        if not rec.get("model_b_markdown"):
            try:
                md = call_iapp_openthai(session, iapp_key, img_path)
                rec["model_b_name"] = "iapp-openthai-2.1"
                rec["model_b_markdown"] = md
                changed = True
            except Exception as exc:
                print(f"  [err] iApp OpenThai 2.1 failed for {sid}: {exc}", flush=True)

        if with_gemini and not rec.get("model_a_markdown"):
            try:
                md, kie = call_vlm_openai_compat(session, KKU_BASE_URL, kku_key, KKU_MODEL_A, img_path)
                rec["model_a_name"] = KKU_MODEL_A
                rec["model_a_markdown"] = md
                rec["model_a_kie"] = kie
                changed = True
            except Exception as exc:
                print(f"  [err] {KKU_MODEL_A} failed for {sid}: {exc}", flush=True)

        if changed:
            if rec.get("model_a_markdown") and rec.get("model_b_markdown"):
                rec["agreement_score"] = compute_agreement(rec["model_a_markdown"], rec["model_b_markdown"])
            with lock:
                existing[sid] = rec
                RAJAKITCHA_GT_DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with RAJAKITCHA_GT_DRAFTS_PATH.open("w", encoding="utf-8") as f:
                    for r in existing.values():
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                processed += 1
                print(
                    f"[{processed}/{len(todo)}] {sid} ({item['relative_path']}) "
                    f"chars={len(rec.get('model_b_markdown', ''))}",
                    flush=True,
                )
            if delay > 0:
                time.sleep(delay)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, item) for item in todo]
        for fut in as_completed(futures):
            fut.result()

    print(f"\nDone: {len(existing)}/{len(manifest)} rajakitcha drafts stored in {RAJAKITCHA_GT_DRAFTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft GT for Config 6 (rajakitcha) with iApp OpenThai OCR 2.1 API.")
    parser.add_argument("--with-gemini", action="store_true", help="Also draft with KKU gemini-4-hydrogen for dual diff.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N images (0 = all).")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_rajakitcha_drafting(
        with_gemini=args.with_gemini,
        limit=args.limit,
        delay=args.delay,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
