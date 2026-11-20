"""Shared domain rules for Thai Saraban PII screening, PUA detection, quality tags, and headers."""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from train_ocr.common.settings import PII_TITLE_BLOCKLIST

THAI_DIGITS_MAP = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

THAI_ID_REGEX = re.compile(
    r"(?<![0-9๐-๙])[1-8๑-๘][-\s]?[0-9๐-๙]{4}[-\s]?[0-9๐-๙]{5}[-\s]?[0-9๐-๙]{2}[-\s]?[0-9๐-๙](?![0-9๐-๙])"
)

PUA_MOJIBAKE_REGEX = re.compile(r"[\uF700-\uF71A\uFFFD]")

SIGNATURE_MARKERS = (
    "ขอแสดงความนับถือ",
    "สั่ง ณ วันที่",
    "ประกาศ ณ วันที่",
    "ให้ไว้ ณ วันที่",
    "ลงชื่อ",
)


def normalize_thai_digits(text: str) -> str:
    """Convert Thai numerals (๐-๙) to Arabic digits (0-9)."""
    return text.translate(THAI_DIGITS_MAP)


def is_valid_thai_id_checksum(digits13: str) -> bool:
    """Validate 13-digit Thai Citizen ID using standard Mod-11 checksum."""
    if len(digits13) != 13 or not digits13.isdigit():
        return False
    if digits13[0] not in "12345678":
        return False
    total = sum(int(d) * (13 - i) for i, d in enumerate(digits13[:12]))
    check = (11 - (total % 11)) % 10
    return check == int(digits13[12])


def has_thai_citizen_id(text: str) -> bool:
    """Detect any valid 13-digit Thai Citizen ID in Arabic or Thai numerals."""
    if not text:
        return False
    for match in THAI_ID_REGEX.finditer(text):
        cleaned = re.sub(r"[-\s]", "", normalize_thai_digits(match.group(0)))
        if is_valid_thai_id_checksum(cleaned):
            return True
    return False


def has_pua_mojibake(text: str) -> bool:
    """Detect legacy Thai font Private Use Area glyphs or replacement chars."""
    return bool(PUA_MOJIBAKE_REGEX.search(text or ""))


def is_pii_blocked_title(title: str) -> bool:
    """Return True if document title/subject indicates HR/personnel/welfare PII risk."""
    t = title or ""
    return any(kw in t for kw in PII_TITLE_BLOCKLIST)


def detect_quality_tags(pil_img: Image.Image) -> list[str]:
    """Compute visual difficulty/ink tags from an RGB page image."""
    w, h = pil_img.size
    if w > 800:
        new_h = max(1, int(h * 800 / w))
        sample_img = pil_img.resize((800, new_h), Image.Resampling.BILINEAR)
    else:
        sample_img = pil_img

    arr = np.asarray(sample_img.convert("RGB"), dtype=np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    tags: list[str] = []
    red_ratio = float(np.mean((r > 150) & ((r - g) > 55) & ((r - b) > 55)))
    if 0.0005 <= red_ratio <= 0.25:
        tags.append("has_red_stamp")

    blue_ratio = float(np.mean((b > 130) & ((b - r) > 45) & ((b - g) > 15)))
    if 0.0005 <= blue_ratio <= 0.25:
        tags.append("has_blue_ink")

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    if float(np.std(gray)) < 25.0:
        tags.append("low_contrast")

    return tags


def detect_header_saraban_type(text: str) -> str | None:
    """Heuristic check of page-1 header text against the 6 Thai Saraban types."""
    if not text:
        return None
    head = " ".join(text[:400].split())
    head_wide = " ".join(text[:1000].split())

    if "บันทึกข้อความ" in head:
        return "internal_memo"
    if "หนังสือประทับตรา" in head:
        return "stamped_letter"
    if re.search(r"ที่\s*[ก-ฮ]", head_wide) and "เรื่อง" in head_wide and "เรียน" in head_wide:
        return "external_letter"
    if re.search(r"(?:รายงานการประชุม|หนังสือรับรอง)", head):
        return "official_record"
    if re.search(r"(?:คำสั่ง|ระเบียบ|ข้อบังคับ)", head):
        return "order_regulation"
    if re.search(r"(?:ประกาศ|แถลงการณ์)", head):
        return "public_relations"
    return None
