"""Backward-compatible shim for train_ocr.collectors.dla."""

from train_ocr.collectors.dla import (
    DLA_QUERY_SPECS,
    collect_dla,
    main,
    parse_dla_rows,
    parse_total_pages,
    pick_best_link,
    post_dla_page,
)
from train_ocr.common.manifest import load_saraban_manifest

__all__ = [
    "DLA_QUERY_SPECS",
    "collect_dla",
    "load_saraban_manifest",
    "main",
    "parse_dla_rows",
    "parse_total_pages",
    "pick_best_link",
    "post_dla_page",
]

if __name__ == "__main__":
    main()
