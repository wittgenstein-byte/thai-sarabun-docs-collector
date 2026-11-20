"""Backward-compatible shim for train_ocr.collectors.datagoth EDA."""

from train_ocr.collectors.datagoth import main_eda as main, run_eda

__all__ = ["main", "run_eda"]

if __name__ == "__main__":
    main()
