"""Shared JSONL manifest reading, writing, and Saraban manifest indexing."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from train_ocr.common.settings import SARABAN_TYPES


def read_jsonl(path: Path) -> list[dict]:
    """Read non-empty JSON lines from a JSONL file (returns empty list if absent)."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record as a UTF-8 line, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Write a sequence of JSON records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_saraban_manifest(
    manifest_path: Path,
) -> tuple[set[str], set[str], dict[str, int], dict[str, int]]:
    """Return (seen_urls, seen_hashes, counts_by_saraban_type, counts_by_subtype)."""
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    type_counts: dict[str, int] = {t: 0 for t in SARABAN_TYPES}
    subtype_counts: dict[str, int] = {}

    for rec in read_jsonl(manifest_path):
        seen_urls.add(rec["source_url"])
        seen_hashes.add(rec["sha256"])
        st = rec["saraban_type"]
        sub = rec.get("saraban_subtype", "")
        type_counts[st] = type_counts.get(st, 0) + 1
        subtype_counts[sub] = subtype_counts.get(sub, 0) + 1

    return seen_urls, seen_hashes, type_counts, subtype_counts
