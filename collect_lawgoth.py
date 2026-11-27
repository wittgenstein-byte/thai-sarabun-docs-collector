"""Backward-compatible shim for train_ocr.collectors.lawgoth."""

from train_ocr.collectors.lawgoth import (
    LawCatalog,
    encode_law_url,
    extract_pdf_files,
    index_lawgoth_pages,
    main,
    run_collect_lawgoth,
)

__all__ = [
    "LawCatalog",
    "encode_law_url",
    "extract_pdf_files",
    "index_lawgoth_pages",
    "main",
    "run_collect_lawgoth",
]

if __name__ == "__main__":
    main()
