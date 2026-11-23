"""Draft Ground Truth for Config 3 (google_docs) using iApp OpenThai OCR 2.1 (+ optional KKU VLM)."""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

from config import (
    IAPP_MARKDOWN_URL,
    KKU_BASE_URL,
    KKU_MODEL_A,
    load_gt_api_keys,
    load_kku_api_key,
)
from draft_gt_dual import call_vlm_openai_compat, compute_agreement
from train_ocr.common.settings import GOOGLE_CRAWL_DIR, GOOGLE_HF_DIR, GOOGLE_PREPROCESSED_DIR

GOOGLE_GT_DIR = GOOGLE_CRAWL_DIR / "gt"
GOOGLE_GT_MANIFEST_PATH = GOOGLE_GT_DIR / "google_manifest.jsonl"
GOOGLE_GT_DRAFTS_PATH = GOOGLE_GT_DIR / "dual_drafts.jsonl"
GOOGLE_GT_ANNOTATIONS_PATH = GOOGLE_GT_DIR / "gt_annotations.jsonl"


def build_google_manifest(
    hf_dir: Path = GOOGLE_HF_DIR,
    preprocessed_dir: Path = GOOGLE_PREPROCESSED_DIR,
    manifest_path: Path = GOOGLE_GT_MANIFEST_PATH,
) -> list[dict]:
    """Generate google_manifest.jsonl from data/google_images/hf_dataset and preprocessed PNGs."""
    from datasets import DatasetDict

    if not hf_dir.exists():
        raise FileNotFoundError(f"Google HF dataset not found: {hf_dir}")

    ds = DatasetDict.load_from_disk(str(hf_dir))["test"]
    records: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        rel = Path(row["relative_path"]).with_suffix(".png")
        img_path = preprocessed_dir / rel
        if not img_path.exists():
            raise FileNotFoundError(f"Preprocessed image missing: {img_path}")
        records.append(
            {
                "sample_id": row["doc_id"],
                "doc_id": row["doc_id"],
                "image_path": str(img_path.as_posix()),
                "relative_path": row["relative_path"],
                "keyword": row["keyword"],
                "saraban_type_hint": row["saraban_type_hint"],
                "saraban_type": row["saraban_type_hint"],
                "page_type": "scanned",
                "page_role": "google_doc",
                "organization": row["keyword"],
                "page_num": 1,
                "total_pages": 1,
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


def _prepare_image_bytes(image_path: Path, max_edge: int = 2400, max_bytes: int = 4_000_000) -> tuple[str, bytes, str]:
    """Return (filename, bytes, mime_type) for iApp upload, resizing only if oversized."""
    raw_bytes = image_path.read_bytes()
    if len(raw_bytes) <= max_bytes:
        return image_path.name, raw_bytes, "image/png"

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size
        if max(w, h) > max_edge:
            scale = max_edge / max(w, h)
            rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=92, optimize=True)
        return image_path.with_suffix(".jpg").name, buf.getvalue(), "image/jpeg"


def call_iapp_openthai(
    session: requests.Session,
    api_key: str,
    image_path: Path,
    url: str = IAPP_MARKDOWN_URL,
    timeout: int = 120,
) -> str:
    """Call iApp OpenThai OCR 2.1 Document-to-Markdown API and return markdown text."""
    fname, img_bytes, mime = _prepare_image_bytes(image_path)
    for attempt in range(3):
        try:
            resp = session.post(
                url,
                headers={"apikey": api_key},
                files={"file": (fname, img_bytes, mime)},
                timeout=timeout,
            )
            if resp.status_code in {429, 502, 503, 504} and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in ctype or resp.text.lstrip().startswith("{"):
                data = resp.json()
                for key in ("markdown", "text", "content", "result", "data"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                    if isinstance(val, dict) and isinstance(val.get("markdown"), str):
                        return val["markdown"].strip()
                return json.dumps(data, ensure_ascii=False)
            return resp.text.strip()
        except requests.Timeout:
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    return ""


def run_google_drafting(
    with_gemini: bool = False,
    limit: int = 0,
    delay: float = 0.2,
    workers: int = 4,
) -> None:
    """Build manifest (if needed) and draft GT for google_docs via iApp OpenThai OCR 2.1."""
    _, iapp_key = load_gt_api_keys()
    if not iapp_key:
        raise ValueError(
            "IAPP_API_KEY (or OPENTHAI_API_KEY) not found in environment or .env. "
            "Please add IAPP_API_KEY=... to .env before running."
        )

    kku_key = load_kku_api_key() if with_gemini else ""

    manifest = build_google_manifest()
    if limit > 0:
        manifest = manifest[:limit]

    existing: dict[str, dict] = {}
    if GOOGLE_GT_DRAFTS_PATH.exists():
        for line in GOOGLE_GT_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
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
        f"Pending google_docs samples: {len(todo)}/{len(manifest)} "
        f"(iApp OpenThai 2.1{' + ' + KKU_MODEL_A if with_gemini else ''}, workers={workers})",
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
                GOOGLE_GT_DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with GOOGLE_GT_DRAFTS_PATH.open("w", encoding="utf-8") as f:
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

    print(f"\nDone: {len(existing)}/{len(manifest)} google_docs drafts stored in {GOOGLE_GT_DRAFTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft GT for Config 3 (google_docs) with iApp OpenThai OCR 2.1.")
    parser.add_argument("--with-gemini", action="store_true", help="Also draft with KKU gemini-4-hydrogen for dual diff.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N images (0 = all).")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_google_drafting(
        with_gemini=args.with_gemini,
        limit=args.limit,
        delay=args.delay,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
