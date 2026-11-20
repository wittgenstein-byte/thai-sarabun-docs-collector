"""Backward-compatible shim for train_ocr.collectors.dla_bulk."""

from train_ocr.collectors.dla_bulk import (
    ALL_SOURCES,
    DlaCatalog,
    main,
    run_bulk_harvest,
)

__all__ = [
    "ALL_SOURCES",
    "DlaCatalog",
    "main",
    "run_bulk_harvest",
]

if __name__ == "__main__":
    main()
