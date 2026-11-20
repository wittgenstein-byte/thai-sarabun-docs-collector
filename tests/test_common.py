"""Unit tests for decoupled common utilities, isolated credentials, and AutoCrawler packaging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import autocrawler
from train_ocr.collectors.google_images import is_blocked_domain
from train_ocr.common.manifest import append_jsonl, load_saraban_manifest, read_jsonl, write_jsonl
from train_ocr.common.saraban_rules import (
    detect_header_saraban_type,
    detect_quality_tags,
    has_pua_mojibake,
    has_thai_citizen_id,
    is_valid_thai_id_checksum,
)
from train_ocr.common.settings import load_credentials, load_datagoth_api_key, load_hf_token


def _make_valid_thai_id(prefix12: str = "110070234567") -> str:
    total = sum(int(d) * (13 - i) for i, d in enumerate(prefix12))
    check = (11 - (total % 11)) % 10
    return prefix12 + str(check)


def test_autocrawler_installed_from_package_not_cloned_dir() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    assert not (repo_root / "data" / "AutoCrawler").exists()
    pkg_path = Path(autocrawler.__file__).resolve()
    assert "site-packages" in str(pkg_path).lower()


def test_thai_id_and_pua_rules() -> None:
    valid_id = _make_valid_thai_id()
    assert is_valid_thai_id_checksum(valid_id)
    assert not is_valid_thai_id_checksum(valid_id[:-1] + str((int(valid_id[-1]) + 1) % 10))
    assert has_thai_citizen_id(f"เลขประจำตัว {valid_id[0]}-{valid_id[1:5]}-{valid_id[5:10]}-{valid_id[10:12]}-{valid_id[12]}")

    thai_digits = valid_id.translate(str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙"))
    assert has_thai_citizen_id(f"บัตรประชาชน {thai_digits}")
    assert not has_thai_citizen_id("เลขหนังสือ ศธ 04025/4201")

    assert has_pua_mojibake("ข้อความ\uf701เสีย")
    assert not has_pua_mojibake("หนังสือราชการปกติ")


def test_saraban_header_and_quality_tags() -> None:
    assert detect_header_saraban_type("บันทึกข้อความ ส่วนราชการ โรงเรียน...") == "internal_memo"
    assert detect_header_saraban_type("ที่ ศธ 04025/4201 เรื่อง ขออนุญาต เรียน ผู้อำนวยการ") == "external_letter"

    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[10:30, 10:30] = [220, 20, 20]  # red stamp
    arr[50:70, 50:70] = [20, 40, 210]  # blue ink
    tags = detect_quality_tags(Image.fromarray(arr))
    assert "has_red_stamp" in tags
    assert "has_blue_ink" in tags


def test_isolated_credentials_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing_env = tmp_path / "missing.env"
    monkeypatch.setenv("DATAGOTH_API_KEY", "dg-secret-only")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("THAI_OCR_HF_TOKEN", raising=False)

    assert load_datagoth_api_key(missing_env) == "dg-secret-only"
    with pytest.raises(FileNotFoundError):
        load_hf_token(missing_env)

    monkeypatch.delenv("DATAGOTH_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf-secret-only")
    assert load_hf_token(missing_env) == "hf-secret-only"
    with pytest.raises(FileNotFoundError):
        load_datagoth_api_key(missing_env)


def test_legacy_env_file_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("DATAGOTH_API_KEY", "DATA_GO_TH_KEY", "HF_TOKEN", "HUGGINGFACE_TOKEN", "THAI_OCR_HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text("bare-datagoth-key\nthai-ocr=hf_abc123\n", encoding="utf-8")
    creds = load_credentials(env_file)
    assert creds.datagoth_api_key == "bare-datagoth-key"
    assert creds.hf_token == "hf_abc123"


def test_manifest_roundtrip_and_blocked_domains(tmp_path: Path) -> None:
    p = tmp_path / "manifest.jsonl"
    rec1 = {
        "source_url": "https://example.go.th/1.pdf",
        "sha256": "abc123",
        "saraban_type": "external_letter",
        "saraban_subtype": "external_circular",
    }
    write_jsonl(p, [rec1])
    append_jsonl(
        p,
        {
            "source_url": "https://example.go.th/2.pdf",
            "sha256": "def456",
            "saraban_type": "internal_memo",
            "saraban_subtype": "memo",
        },
    )
    rows = read_jsonl(p)
    assert len(rows) == 2

    urls, hashes, type_counts, subtype_counts = load_saraban_manifest(p)
    assert urls == {"https://example.go.th/1.pdf", "https://example.go.th/2.pdf"}
    assert hashes == {"abc123", "def456"}
    assert type_counts["external_letter"] == 1
    assert subtype_counts["memo"] == 1

    assert is_blocked_domain("https://www.lemon8-app.com/seo/image?id=1")
    assert is_blocked_domain("https://image.slidesharecdn.com/slide.jpg")
    assert is_blocked_domain("https://www.facebook.com/photo.php?fbid=123")
    assert is_blocked_domain("https://i.pinimg.com/originals/abc.jpg")
    assert is_blocked_domain("https://www.tiktok.com/@user/video/1")
    assert not is_blocked_domain("https://www.kkzone1.go.th/doc.jpg")
