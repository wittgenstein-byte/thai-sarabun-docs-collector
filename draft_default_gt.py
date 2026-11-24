"""Draft Ground Truth for Config 2 (default: data.go.th) using KKU GenAI VLMs."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from config import (
    DATA_DIR,
    HF_DATASET_DIR,
    KKU_BASE_URL,
    KKU_MODEL_A,
    KKU_MODEL_B,
    PAGES_DIR,
    load_kku_api_key,
)
from draft_gt_dual import call_vlm_openai_compat, compute_agreement

DEFAULT_GT_DIR = DATA_DIR / "default_gt"
DEFAULT_GT_MANIFEST_PATH = DEFAULT_GT_DIR / "default_manifest.jsonl"
DEFAULT_GT_DRAFTS_PATH = DEFAULT_GT_DIR / "dual_drafts.jsonl"
DEFAULT_GT_ANNOTATIONS_PATH = DEFAULT_GT_DIR / "gt_annotations.jsonl"


def build_default_manifest(
    hf_dir: Path = HF_DATASET_DIR,
    pages_dir: Path = PAGES_DIR,
    manifest_path: Path = DEFAULT_GT_MANIFEST_PATH,
    target: int = 120,
) -> list[dict]:
    """Generate stratified default_manifest.jsonl (up to target pages) from data/hf_dataset."""
    from collections import defaultdict
    from datasets import DatasetDict

    if not hf_dir.exists():
        raise FileNotFoundError(f"Default HF dataset not found: {hf_dir}")

    drafted_ids: set[str] = set()
    if DEFAULT_GT_DRAFTS_PATH.exists():
        for line in DEFAULT_GT_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                drafted_ids.add(json.loads(line)["sample_id"])

    ds = DatasetDict.load_from_disk(str(hf_dir))["test"]
    all_records: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        sid = f"{row['resource_id']}_p{row['page_num']}"
        img_path = pages_dir / row["group"] / f"{sid}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Page image missing: {img_path}")
        all_records.append(
            {
                "sample_id": sid,
                "doc_id": row["doc_id"],
                "resource_id": row["resource_id"],
                "image_path": str(img_path.as_posix()),
                "group": row["group"],
                "saraban_type": row["group"],
                "page_type": row["page_type"],
                "page_role": row.get("title") or row["group"],
                "organization": row["organization"],
                "title": row["title"],
                "page_num": int(row["page_num"]),
                "total_pages": int(row["total_pages"]),
                "embedded_text_preview": row.get("embedded_text_preview") or "",
                "width": int(row["width"]),
                "height": int(row["height"]),
                "quality_tags": list(row.get("quality_tags") or []),
            }
        )

    if target > 0 and target < len(all_records):
        by_group: dict[str, list[dict]] = defaultdict(list)
        for rec in all_records:
            by_group[rec["group"]].append(rec)
        # Prioritize already-drafted pages first within each group, then diverse resource_id
        for grp in by_group:
            by_group[grp].sort(key=lambda r: (0 if r["sample_id"] in drafted_ids else 1, r["page_type"], r["sample_id"]))
        selected: list[dict] = []
        groups = sorted(by_group)
        while len(selected) < target and any(by_group[g] for g in groups):
            for grp in groups:
                if by_group[grp] and len(selected) < target:
                    selected.append(by_group[grp].pop(0))
        records = selected
    else:
        records = all_records

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def run_default_drafting(
    base_url: str = KKU_BASE_URL,
    model_a: str = KKU_MODEL_A,
    model_b: str = KKU_MODEL_B,
    only: str = "both",
    target: int = 120,
    limit: int = 0,
    delay: float = 0.1,
    workers: int = 6,
) -> None:
    """Build manifest and generate dual-model KKU VLM drafts for default dataset."""
    api_key = load_kku_api_key()
    manifest = build_default_manifest(target=target)
    if limit > 0:
        manifest = manifest[:limit]

    existing: dict[str, dict] = {}
    if DEFAULT_GT_DRAFTS_PATH.exists():
        for line in DEFAULT_GT_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
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
        f"Pending default samples: {len(todo)}/{len(manifest)} "
        f"(workers={workers}, models={model_a} + {model_b}, only={only})",
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
                print(f"  [err] {model_a} failed for {sid}: {exc}", flush=True)

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
                DEFAULT_GT_DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with DEFAULT_GT_DRAFTS_PATH.open("w", encoding="utf-8") as f:
                    for r in existing.values():
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                processed += 1
                print(
                    f"[{processed}/{len(todo)}] {sid} ({item['group']}/{item['page_type']}) "
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
        f"\nDone: {len(existing)}/{len(manifest)} default drafts stored in {DEFAULT_GT_DRAFTS_PATH}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft GT for Config 2 (default) via KKU GenAI VLMs.")
    parser.add_argument("--base-url", type=str, default=KKU_BASE_URL)
    parser.add_argument("--model-a", type=str, default=KKU_MODEL_A)
    parser.add_argument("--model-b", type=str, default=KKU_MODEL_B)
    parser.add_argument("--only", choices=["both", "a", "b"], default="both")
    parser.add_argument("--target", type=int, default=120, help="Stratified subset size (0 = all 319).")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N pages (0 = all).")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run_default_drafting(
        base_url=args.base_url,
        model_a=args.model_a,
        model_b=args.model_b,
        only=args.only,
        target=args.target,
        limit=args.limit,
        delay=args.delay,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
