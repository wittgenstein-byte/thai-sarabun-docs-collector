"""Backward-compatible shim for train_ocr.collectors.datagoth general collector."""

from train_ocr.collectors.datagoth import (
    collect_general as collect,
    download_and_validate_general_pdf as download_and_validate_pdf,
    fetch_group_candidates,
    has_scanned_pages,
    load_existing_manifest,
    main_general as main,
)

__all__ = [
    "collect",
    "download_and_validate_pdf",
    "fetch_group_candidates",
    "has_scanned_pages",
    "load_existing_manifest",
    "main",
]

if __name__ == "__main__":
    main()
