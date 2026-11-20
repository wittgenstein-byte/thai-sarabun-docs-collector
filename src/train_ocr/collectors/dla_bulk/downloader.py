"""Concurrent, resumable file downloader with SHA-256 deduplication and budget cap."""

from __future__ import annotations

import hashlib
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

from train_ocr.collectors.dla_bulk.catalog import DlaCatalog
from train_ocr.collectors.dla_bulk.portals import make_session

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".gif", ".bmp"}


def _ext_from_content_type(content_type: str) -> str:
    ct = content_type.split(";")[0].strip().lower()
    mapping = {
        "application/pdf": ".pdf",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/tiff": ".tiff",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.ms-excel": ".xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/zip": ".zip",
    }
    return mapping.get(ct, "")


def _validate_content(raw: bytes, ext: str, allowed_exts: tuple[str, ...] | None) -> tuple[bool, int, str]:
    """Validate file bytes against magic headers and allowed extensions.

    Returns (is_valid, total_pages_for_pdf, reason_if_invalid).
    """
    norm_ext = ext.lower()
    if raw.lstrip()[:5] == b"%PDF-":
        norm_ext = ".pdf"

    if allowed_exts and norm_ext not in allowed_exts:
        return False, 0, f"disallowed_ext:{norm_ext or 'unknown'}"

    if norm_ext == ".pdf":
        if not raw.lstrip().startswith(b"%PDF-"):
            return False, 0, "invalid_pdf_magic"
        try:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(raw)
            try:
                num_pages = len(doc)
            finally:
                doc.close()
            if num_pages < 1:
                return False, 0, "empty_pdf"
            return True, num_pages, ""
        except Exception as exc:
            return False, 0, f"corrupt_pdf:{type(exc).__name__}"

    if norm_ext in IMAGE_EXTS:
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
            return True, 1, ""
        except Exception as exc:
            return False, 0, f"corrupt_image:{type(exc).__name__}"

    return True, 0, ""


def run_downloader(
    catalog: DlaCatalog,
    files_dir: Path,
    *,
    sources: list[str] | None = None,
    allowed_exts: tuple[str, ...] | None = None,
    workers: int = 4,
    delay: float = 0.15,
    timeout: int = 30,
    min_bytes: int = 1024,
    max_file_bytes: int = 50 * 1024 * 1024,
    max_total_bytes: int = 1024 * 1024 * 1024,
    max_files: int | None = None,
    retry_failed: bool = False,
    batch_size: int = 100,
) -> dict[str, int]:
    """Download pending files concurrently until queue is empty or limits are reached."""
    files_dir = Path(files_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    thread_local = threading.local()
    budget_lock = threading.Lock()
    current_total_bytes = catalog.get_total_downloaded_bytes()
    stats = {
        "downloaded": 0,
        "duplicate": 0,
        "skipped": 0,
        "failed": 0,
        "bytes_added": 0,
    }

    if max_total_bytes > 0 and current_total_bytes >= max_total_bytes:
        print(
            f"[download] Storage budget already reached "
            f"({current_total_bytes / (1024**2):.1f} MB / {max_total_bytes / (1024**2):.1f} MB)."
        )
        return stats

    def get_session() -> requests.Session:
        sess = getattr(thread_local, "session", None)
        if sess is None:
            sess = make_session(pool_maxsize=max(8, workers * 2))
            thread_local.session = sess
        return sess

    def download_one(item: dict) -> tuple[str, int]:
        nonlocal current_total_bytes
        url = item["url"]
        source = item["source"]
        ext = (item["ext"] or "").lower()

        with budget_lock:
            if max_total_bytes > 0 and current_total_bytes >= max_total_bytes:
                return "budget_reached", 0

        if delay > 0:
            time.sleep(delay)

        sess = get_session()
        try:
            with sess.get(url, stream=True, allow_redirects=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    if resp.status_code in (404, 410):
                        catalog.mark_skipped(url, f"http_{resp.status_code}", http_status=resp.status_code)
                        return "skipped", 0
                    catalog.mark_failed(url, f"http_{resp.status_code}", http_status=resp.status_code)
                    return "failed", 0

                if not ext:
                    ext = _ext_from_content_type(resp.headers.get("Content-Type", ""))

                clen = resp.headers.get("Content-Length")
                if clen and clen.isdigit() and int(clen) > max_file_bytes:
                    catalog.mark_skipped(url, f"too_large:{clen}", http_status=200)
                    return "skipped", 0

                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_file_bytes:
                        catalog.mark_skipped(url, f"too_large:>{max_file_bytes}", http_status=200)
                        return "skipped", 0
                    chunks.append(chunk)

                if total < min_bytes:
                    catalog.mark_skipped(url, f"too_small:{total}", http_status=200)
                    return "skipped", 0

                raw = b"".join(chunks)
        except requests.RequestException as exc:
            catalog.mark_failed(url, str(exc))
            return "failed", 0

        if raw.lstrip()[:5] == b"%PDF-":
            ext = ".pdf"
        if not ext:
            ext = ".bin"

        valid, total_pages, reason = _validate_content(raw, ext, allowed_exts)
        if not valid:
            catalog.mark_skipped(url, reason, http_status=200)
            return "skipped", 0

        sha256 = hashlib.sha256(raw).hexdigest()
        existing_path = catalog.has_sha256(sha256)
        if existing_path:
            catalog.mark_duplicate(
                url,
                sha256=sha256,
                size_bytes=total,
                existing_path=existing_path,
                ext=ext,
                http_status=200,
            )
            return "duplicate", 0

        with budget_lock:
            if max_total_bytes > 0 and current_total_bytes + total > max_total_bytes:
                return "budget_reached", 0
            current_total_bytes += total

        bucket_dir = files_dir / source / sha256[:2]
        bucket_dir.mkdir(parents=True, exist_ok=True)
        dest_path = bucket_dir / f"{sha256[:16]}{ext}"
        part_path = bucket_dir / f"{sha256[:16]}{ext}.part"
        part_path.write_bytes(raw)
        part_path.replace(dest_path)

        catalog.mark_downloaded(
            url,
            sha256=sha256,
            size_bytes=total,
            total_pages=total_pages,
            local_path=str(dest_path.as_posix()),
            ext=ext,
            http_status=200,
        )
        return "downloaded", total

    processed_success = 0
    budget_hit = False

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while not budget_hit:
            if max_files is not None and processed_success >= max_files:
                break

            fetch_limit = batch_size
            if max_files is not None:
                fetch_limit = min(batch_size, max_files - processed_success)

            batch = catalog.fetch_pending_files(
                limit=fetch_limit,
                sources=sources,
                allowed_exts=allowed_exts,
                include_failed=retry_failed,
            )
            if not batch:
                break

            futures = [pool.submit(download_one, item) for item in batch]
            for fut in as_completed(futures):
                outcome, added_bytes = fut.result()
                if outcome == "budget_reached":
                    budget_hit = True
                    continue
                stats[outcome] = stats.get(outcome, 0) + 1
                if outcome == "downloaded":
                    processed_success += 1
                    stats["bytes_added"] += added_bytes

            mb_total = current_total_bytes / (1024 * 1024)
            mb_cap = max_total_bytes / (1024 * 1024) if max_total_bytes > 0 else 0.0
            print(
                f"  [download] +{stats['downloaded']} saved | {stats['duplicate']} dup | "
                f"{stats['skipped']} skipped | {stats['failed']} failed | "
                f"total={mb_total:.1f}/{mb_cap:.0f} MB"
            )

    return stats
