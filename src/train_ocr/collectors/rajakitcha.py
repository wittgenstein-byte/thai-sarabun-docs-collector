"""Collector for Rajakitcha (Royal Gazette) document images from GitHub."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from train_ocr.common.settings import (
    RAJAKITCHA_MANIFEST_PATH,
    RAJAKITCHA_RAW_DIR,
    RAJAKITCHA_SOURCE_PATH,
    RAJAKITCHA_SOURCE_REF,
    RAJAKITCHA_SOURCE_REPO,
    REQUEST_TIMEOUT,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "train-ocr-collector",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def list_github_directory(
    repo: str = RAJAKITCHA_SOURCE_REPO,
    path: str = RAJAKITCHA_SOURCE_PATH,
    ref: str = RAJAKITCHA_SOURCE_REF,
    timeout: int = REQUEST_TIMEOUT,
) -> list[dict[str, Any]]:
    """List image files in a GitHub repository directory via the Contents API."""
    url = f"https://api.github.com/repos/{repo}/contents/{path.strip('/')}"
    resp = requests.get(
        url,
        params={"ref": ref},
        headers=_github_headers(),
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        raise ValueError(f"Expected directory listing from {url}, got {type(payload).__name__}")

    items = [
        item
        for item in payload
        if isinstance(item, dict)
        and item.get("type") == "file"
        and Path(item.get("name", "")).suffix.lower() in IMAGE_EXTS
        and item.get("download_url")
    ]
    items.sort(key=lambda x: x["name"])
    return items


def _download_one(
    item: dict[str, Any],
    raw_dir: Path,
    force: bool = False,
    timeout: int = REQUEST_TIMEOUT,
) -> dict[str, Any]:
    name = item["name"]
    dest = raw_dir / name
    expected_size = int(item.get("size") or 0)

    if not force and dest.exists() and (expected_size == 0 or dest.stat().st_size == expected_size):
        status = "skipped_existing"
    else:
        resp = requests.get(item["download_url"], headers={"User-Agent": "train-ocr-collector"}, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        status = "downloaded"

    return {
        "filename": name,
        "category": "ราชกิจจา",
        "source_repo": RAJAKITCHA_SOURCE_REPO,
        "source_path": item.get("path", ""),
        "git_sha": item.get("sha", ""),
        "size": dest.stat().st_size,
        "download_url": item["download_url"],
        "html_url": item.get("html_url", ""),
        "local_path": str(dest),
        "status": status,
    }


def collect_rajakitcha(
    raw_dir: Path = RAJAKITCHA_RAW_DIR,
    manifest_path: Path = RAJAKITCHA_MANIFEST_PATH,
    repo: str = RAJAKITCHA_SOURCE_REPO,
    path: str = RAJAKITCHA_SOURCE_PATH,
    ref: str = RAJAKITCHA_SOURCE_REF,
    workers: int = 8,
    force: bool = False,
) -> dict[str, Any]:
    """Download Rajakitcha images from GitHub and write manifest.jsonl."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    items = list_github_directory(repo=repo, path=path, ref=ref)
    if not items:
        raise RuntimeError(f"No image files found at {repo}/{path}@{ref}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_download_one, item, raw_dir, force) for item in items]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r["filename"])
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    downloaded = sum(1 for r in results if r["status"] == "downloaded")
    skipped = sum(1 for r in results if r["status"] == "skipped_existing")
    total_bytes = sum(int(r["size"]) for r in results)
    summary = {
        "total_files": len(results),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "total_bytes": total_bytes,
        "raw_dir": str(raw_dir),
        "manifest_path": str(manifest_path),
    }
    print(
        f"Collected Rajakitcha: {summary['total_files']} files "
        f"(downloaded={downloaded}, skipped={skipped}, {total_bytes / (1024 * 1024):.2f} MB) -> {raw_dir}"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Collect Rajakitcha (Royal Gazette) document images from GitHub."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAJAKITCHA_RAW_DIR)
    parser.add_argument("--manifest-path", type=Path, default=RAJAKITCHA_MANIFEST_PATH)
    parser.add_argument("--repo", default=RAJAKITCHA_SOURCE_REPO)
    parser.add_argument("--path", default=RAJAKITCHA_SOURCE_PATH)
    parser.add_argument("--ref", default=RAJAKITCHA_SOURCE_REF)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args(argv)
    collect_rajakitcha(
        raw_dir=args.raw_dir,
        manifest_path=args.manifest_path,
        repo=args.repo,
        path=args.path,
        ref=args.ref,
        workers=args.workers,
        force=args.force,
    )
