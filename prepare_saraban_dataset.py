"""Backward-compatible shim for train_ocr.dataset.builder (Saraban)."""

from train_ocr.dataset.builder import (
    ROLE_PRIORITY,
    build_saraban_dataset,
    main_saraban as main,
    order_candidates_diverse,
)

__all__ = [
    "ROLE_PRIORITY",
    "build_saraban_dataset",
    "main",
    "order_candidates_diverse",
]

if __name__ == "__main__":
    main()
