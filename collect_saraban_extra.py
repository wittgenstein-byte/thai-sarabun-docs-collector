"""Backward-compatible shim for train_ocr.collectors.datagoth Saraban extra collector."""

from train_ocr.collectors.datagoth import (
    DATA_GOTH_QUERIES,
    collect_saraban_extra as collect_extra,
    main_saraban_extra as main,
)

__all__ = ["DATA_GOTH_QUERIES", "collect_extra", "main"]

if __name__ == "__main__":
    main()
