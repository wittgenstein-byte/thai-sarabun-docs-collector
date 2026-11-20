"""Shared PDF downloading, validation, and page-sampling utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import requests

from train_ocr.common.settings import (
    IMAGE_COVERAGE_THRESHOLD,
    MAX_PDF_BYTES,
    MIN_PDF_BYTES,
    REQUEST_TIMEOUT,
)


def _get_pdfium():
    import pypdfium2 as pdfium

    return pdfium


def compute_image_coverage(page: Any) -> float:
    """Compute fraction of page area covered by embedded raster images (0.0 - 1.0)."""
    pdfium = _get_pdfium()
    w, h = page.get_size()
    page_area = max(w * h, 1.0)
    img_area = 0.0
    try:
        for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
            left, bottom, right, top = obj.get_bounds()
            cw = max(0.0, min(right, w) - max(left, 0.0))
            ch = max(0.0, min(top, h) - max(bottom, 0.0))
            img_area += cw * ch
    except Exception:
        return 0.0
    return min(1.0, img_area / page_area)


def pick_scan_indices(total_pages: int, max_scan: int) -> list[int]:
    """Select up to max_scan evenly spaced 0-based page indices."""
    if total_pages <= max_scan:
        return list(range(total_pages))
    step = (total_pages - 1) / (max_scan - 1)
    return sorted({min(total_pages - 1, round(i * step)) for i in range(max_scan)})


def has_scanned_pages(
    doc: Any,
    max_check: int = 10,
    image_coverage_threshold: float = IMAGE_COVERAGE_THRESHOLD,
) -> tuple[int, int]:
    """Return (scanned_count, scanned_ocr_count) among sampled pages."""
    scanned = 0
    scanned_ocr = 0
    for idx in pick_scan_indices(len(doc), max_check):
        page = doc[idx]
        try:
            cov = compute_image_coverage(page)
            if cov > image_coverage_threshold:
                tp = page.get_textpage()
                try:
                    chars = len((tp.get_text_range() or "").strip())
                finally:
                    tp.close()
                if chars < 20:
                    scanned += 1
                else:
                    scanned_ocr += 1
        finally:
            page.close()
    return scanned, scanned_ocr


def download_and_validate_pdf(
    session: requests.Session,
    url: str,
    dest_path: Path | None = None,
    dest_dir: Path | None = None,
    scanned_only: bool = False,
    timeout: int = REQUEST_TIMEOUT,
    check_head: bool = False,
) -> tuple[bool, str, int, int, Path | None]:
    """Download and validate a PDF against size, magic header, and pypdfium2 openability.

    Provide either ``dest_path`` (explicit target path) or ``dest_dir`` (writes ``{sha256[:16]}.pdf``).
    Returns ``(ok, sha256, size_bytes, total_pages, saved_path)``.
    """
    if dest_path is None and dest_dir is None:
        raise ValueError("Either dest_path or dest_dir must be provided")

    if check_head:
        try:
            head = session.head(url, allow_redirects=True, timeout=12)
            if head.status_code == 200:
                clen = head.headers.get("Content-Length")
                if clen and clen.isdigit():
                    size_est = int(clen)
                    if size_est > MAX_PDF_BYTES or size_est < MIN_PDF_BYTES:
                        return False, "", 0, 0, None
        except requests.RequestException:
            pass

    target_path: Path | None = dest_path
    created_new = False
    try:
        with session.get(url, stream=True, allow_redirects=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return False, "", 0, 0, None

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    return False, "", 0, 0, None
                chunks.append(chunk)

            if total < MIN_PDF_BYTES:
                return False, "", 0, 0, None

            raw = b"".join(chunks)
            if not raw.lstrip().startswith(b"%PDF-"):
                return False, "", 0, 0, None

            sha256 = hashlib.sha256(raw).hexdigest()
            if target_path is None:
                assert dest_dir is not None
                dest_dir.mkdir(parents=True, exist_ok=True)
                target_path = dest_dir / f"{sha256[:16]}.pdf"
            else:
                target_path.parent.mkdir(parents=True, exist_ok=True)

            created_new = not target_path.exists()
            target_path.write_bytes(raw)

            pdfium = _get_pdfium()
            doc = pdfium.PdfDocument(str(target_path))
            try:
                num_pages = len(doc)
                if num_pages < 1:
                    target_path.unlink(missing_ok=True)
                    return False, "", 0, 0, None
                if scanned_only:
                    sc, sc_ocr = has_scanned_pages(doc)
                    if sc + sc_ocr == 0:
                        target_path.unlink(missing_ok=True)
                        return False, "", 0, 0, None
            finally:
                doc.close()

            return True, sha256, total, num_pages, target_path
    except Exception:
        if target_path is not None and created_new:
            target_path.unlink(missing_ok=True)
        return False, "", 0, 0, None
