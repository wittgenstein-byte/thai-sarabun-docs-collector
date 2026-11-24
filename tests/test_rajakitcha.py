"""Tests for the Rajakitcha collector and dataset builder."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from datasets import DatasetDict
from PIL import Image

from train_ocr.collectors.rajakitcha import collect_rajakitcha, list_github_directory
from train_ocr.dataset.builder import build_rajakitcha_dataset


def _make_png_bytes(color: tuple[int, int, int] = (255, 255, 255), size: tuple[int, int] = (120, 160)) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_list_github_directory_filters_images() -> None:
    payload = [
        {"name": "page1.png", "type": "file", "download_url": "https://example.com/page1.png", "size": 100},
        {"name": "page0.jpg", "type": "file", "download_url": "https://example.com/page0.jpg", "size": 200},
        {"name": "README.md", "type": "file", "download_url": "https://example.com/README.md", "size": 50},
        {"name": "subdir", "type": "dir", "download_url": None},
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status.return_value = None

    with patch("train_ocr.collectors.rajakitcha.requests.get", return_value=mock_resp):
        items = list_github_directory(repo="owner/repo", path="Example/Constituencies", ref="main")

    assert [i["name"] for i in items] == ["page0.jpg", "page1.png"]


def test_collect_rajakitcha_downloads_and_skips_existing(tmp_path: Path) -> None:
    png_a = _make_png_bytes((240, 240, 240))
    png_b = _make_png_bytes((200, 200, 200))

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    # Pre-populate page0.png with exact matching size so it should be skipped
    (raw_dir / "page0.png").write_bytes(png_a)
    manifest_path = tmp_path / "manifest.jsonl"

    items = [
        {
            "name": "page0.png",
            "type": "file",
            "path": "Example/Constituencies/page0.png",
            "sha": "sha0",
            "size": len(png_a),
            "download_url": "https://raw.example.com/page0.png",
            "html_url": "https://github.com/owner/repo/blob/main/page0.png",
        },
        {
            "name": "page1.png",
            "type": "file",
            "path": "Example/Constituencies/page1.png",
            "sha": "sha1",
            "size": len(png_b),
            "download_url": "https://raw.example.com/page1.png",
            "html_url": "https://github.com/owner/repo/blob/main/page1.png",
        },
    ]

    mock_dl = MagicMock()
    mock_dl.content = png_b
    mock_dl.raise_for_status.return_value = None

    with (
        patch("train_ocr.collectors.rajakitcha.list_github_directory", return_value=items),
        patch("train_ocr.collectors.rajakitcha.requests.get", return_value=mock_dl) as mock_get,
    ):
        summary = collect_rajakitcha(raw_dir=raw_dir, manifest_path=manifest_path, workers=2)

    assert summary["total_files"] == 2
    assert summary["downloaded"] == 1
    assert summary["skipped_existing"] == 1
    mock_get.assert_called_once()
    assert (raw_dir / "page1.png").read_bytes() == png_b

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(r["category"] == "ราชกิจจา" for r in rows)


def test_build_rajakitcha_dataset_deduplicates_and_builds_hf(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    preprocessed_dir = tmp_path / "preprocessed"
    hf_dir = tmp_path / "hf_dataset"

    png_a = _make_png_bytes((250, 250, 250))
    png_b = _make_png_bytes((220, 220, 220))
    (raw_dir / "page0.png").write_bytes(png_a)
    (raw_dir / "page0_dup.png").write_bytes(png_a)
    (raw_dir / "page1.png").write_bytes(png_b)

    with patch("train_ocr.dataset.builder.RAJAKITCHA_DIR", tmp_path):
        stats = build_rajakitcha_dataset(
            raw_dir=raw_dir,
            preprocessed_dir=preprocessed_dir,
            hf_dir=hf_dir,
            max_edge=3508,
        )

    assert stats["config_name"] == "rajakitcha"
    assert stats["total_pages"] == 2
    assert stats["duplicates_skipped"] == 1
    assert (tmp_path / "dataset_stats.json").exists()

    ds_dict = DatasetDict.load_from_disk(str(hf_dir))
    assert "train" in ds_dict
    assert len(ds_dict["train"]) == 2
    row = ds_dict["train"][0]
    assert row["category"] == "ราชกิจจา"
    assert row["subcategory"] == "constituencies"
    assert row["gt_status"] == "unverified"
