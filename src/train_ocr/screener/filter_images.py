"""Batch CLI to filter crawled Google Images into kept/ and rejected/ directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from train_ocr.common.settings import GOOGLE_CRAWL_DIR
from train_ocr.screener.client import ScreenerClient
from train_ocr.screener.engine import IMAGE_EXTS


def filter_directory(
    input_dir: Path,
    output_dir: Path,
    min_dim: int = 800,
    min_boxes: int = 8,
    min_chars: int = 60,
    screener_url: str | None = None,
) -> list[dict]:
    """Screen all images under input_dir, deduplicate by SHA256, and copy into output_dir/{kept,rejected}."""
    images = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        print(f"No images found under {input_dir}")
        return []

    kept_dir = output_dir / "kept"
    rejected_dir = output_dir / "rejected"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    kept_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    client = ScreenerClient(base_url=screener_url)
    records: list[dict] = []
    seen_hashes: dict[str, str] = {}

    for img_path in images:
        rel = img_path.relative_to(input_dir)
        rel_posix = str(rel.as_posix())
        keyword = img_path.parent.name if img_path.parent != input_dir else ""
        raw_bytes = img_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()

        if sha256 in seen_hashes:
            rec = {
                "file": str(img_path),
                "name": img_path.name,
                "keyword": keyword,
                "relative_path": rel_posix,
                "sha256": sha256,
                "size_bytes": len(raw_bytes),
                "status": "rejected",
                "reasons": [f"duplicate({seen_hashes[sha256]})"],
            }
        else:
            seen_hashes[sha256] = rel_posix
            try:
                with Image.open(img_path) as raw:
                    result = client.screen_document(
                        raw,
                        min_dim=min_dim,
                        min_boxes=min_boxes,
                        min_chars=min_chars,
                    )
                rec = {
                    "file": str(img_path),
                    "name": img_path.name,
                    "keyword": keyword,
                    "relative_path": rel_posix,
                    "sha256": sha256,
                    "size_bytes": len(raw_bytes),
                    **result,
                }
            except Exception as exc:
                rec = {
                    "file": str(img_path),
                    "name": img_path.name,
                    "keyword": keyword,
                    "relative_path": rel_posix,
                    "sha256": sha256,
                    "size_bytes": len(raw_bytes),
                    "status": "rejected",
                    "reasons": [f"unreadable({exc})"],
                }

        records.append(rec)
        dest_base = kept_dir if rec["status"] == "kept" else rejected_dir
        dest_path = dest_base / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_path)

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    by_kw: dict[str, dict[str, int]] = defaultdict(lambda: {"kept": 0, "rejected": 0, "duplicate": 0})
    for r in records:
        kw = r.get("keyword") or "(root)"
        if r["status"] == "kept":
            by_kw[kw]["kept"] += 1
        else:
            by_kw[kw]["rejected"] += 1
            if any(reason.startswith("duplicate(") for reason in r.get("reasons", [])):
                by_kw[kw]["duplicate"] += 1

    kept = [r for r in records if r["status"] == "kept"]
    rejected = [r for r in records if r["status"] == "rejected"]
    dup_total = sum(v["duplicate"] for v in by_kw.values())

    print(f"Total: {len(records)} | Kept: {len(kept)} | Rejected: {len(rejected)} (Duplicates: {dup_total})")
    print("\n--- Per-Keyword Summary ---")
    for kw, counts in sorted(by_kw.items()):
        print(
            f"  {kw}: kept={counts['kept']}, rejected={counts['rejected']} "
            f"(duplicates={counts['duplicate']})"
        )

    print("\n--- Item Details ---")
    for r in records:
        tag = "KEEP  " if r["status"] == "kept" else "REJECT"
        dims = f"{r.get('width', '?')}x{r.get('height', '?')}"
        boxes = r.get("ocr_boxes", "?")
        chars = r.get("ocr_chars", "?")
        reasons = ", ".join(r.get("reasons", [])) or "-"
        print(f"[{tag}] {r['relative_path']} | {dims} | boxes={boxes} chars={chars} | {reasons}")

    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Filter crawled images for OCR document quality.")
    parser.add_argument("--input-dir", type=Path, default=GOOGLE_CRAWL_DIR / "raw")
    parser.add_argument("--output-dir", type=Path, default=GOOGLE_CRAWL_DIR / "filtered")
    parser.add_argument("--min-dim", type=int, default=800)
    parser.add_argument("--min-boxes", type=int, default=8)
    parser.add_argument("--min-chars", type=int, default=60)
    parser.add_argument("--screener-url", type=str, default=None)
    args = parser.parse_args(argv)
    filter_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_dim=args.min_dim,
        min_boxes=args.min_boxes,
        min_chars=args.min_chars,
        screener_url=args.screener_url,
    )
