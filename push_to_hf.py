"""Backward-compatible shim for train_ocr.publisher.hf_hub."""

from train_ocr.publisher.hf_hub import (
    build_combined_readme,
    extract_yaml_frontmatter,
    main,
    merge_hub_yaml,
    push_dataset,
)

__all__ = [
    "build_combined_readme",
    "extract_yaml_frontmatter",
    "main",
    "merge_hub_yaml",
    "push_dataset",
]

if __name__ == "__main__":
    main()
