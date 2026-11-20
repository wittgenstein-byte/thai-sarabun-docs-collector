"""Backward-compatible shim for train_ocr.dataset.builder (general)."""

from train_ocr.dataset.builder import (
    build_general_dataset as build_dataset,
    main_general as main,
    round_robin_select,
)

__all__ = ["build_dataset", "main", "round_robin_select"]

if __name__ == "__main__":
    main()
