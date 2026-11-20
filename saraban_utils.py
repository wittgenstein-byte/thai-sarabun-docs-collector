"""Backward-compatible shim for train_ocr.common.saraban_rules."""

from train_ocr.common.saraban_rules import (
    PUA_MOJIBAKE_REGEX,
    SIGNATURE_MARKERS,
    THAI_DIGITS_MAP,
    THAI_ID_REGEX,
    detect_header_saraban_type,
    detect_quality_tags,
    has_pua_mojibake,
    has_thai_citizen_id,
    is_pii_blocked_title,
    is_valid_thai_id_checksum,
    normalize_thai_digits,
)

__all__ = [
    "PUA_MOJIBAKE_REGEX",
    "SIGNATURE_MARKERS",
    "THAI_DIGITS_MAP",
    "THAI_ID_REGEX",
    "detect_header_saraban_type",
    "detect_quality_tags",
    "has_pua_mojibake",
    "has_thai_citizen_id",
    "is_pii_blocked_title",
    "is_valid_thai_id_checksum",
    "normalize_thai_digits",
]
