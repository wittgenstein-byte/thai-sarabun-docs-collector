"""Backward-compatible shim for train_ocr.screener.pdf_scanner (general)."""

from train_ocr.common.pdf_io import compute_image_coverage, pick_scan_indices
from train_ocr.screener.pdf_scanner import (
    analyze_page,
    main_general as main,
    scan_general_all as scan_all,
)

__all__ = [
    "analyze_page",
    "compute_image_coverage",
    "main",
    "pick_scan_indices",
    "scan_all",
]

if __name__ == "__main__":
    main()
