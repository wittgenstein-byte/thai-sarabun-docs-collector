"""Backward-compatible shim for train_ocr.screener.pdf_scanner (Saraban)."""

from train_ocr.screener.pdf_scanner import (
    extract_full_text,
    find_signature_page_idx,
    main_saraban as main,
    scan_saraban_all,
)

__all__ = [
    "extract_full_text",
    "find_signature_page_idx",
    "main",
    "scan_saraban_all",
]

if __name__ == "__main__":
    main()
