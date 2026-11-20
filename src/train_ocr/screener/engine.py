"""Shared RapidOCR + Pillow screening engine for PII detection and document image filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageStat

from train_ocr.common.saraban_rules import (
    detect_header_saraban_type,
    detect_quality_tags,
    has_thai_citizen_id,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
GOV_MARKERS = (
    "เรื่อง",
    "เรียน",
    "ที่",
    "บันทึกข้อความ",
    "คำสั่ง",
    "ประกาศ",
    "ส่วนราชการ",
    "โรงเรียน",
    "สำนักงาน",
    "ลงชื่อ",
    "พ.ศ.",
    "ราชการ",
    "หนังสือ",
)


class OCRScreener:
    """Wraps RapidOCR (loaded once) and Pillow heuristics for PII and document screening."""

    def __init__(self, ocr_engine: Any | None = None) -> None:
        if ocr_engine is None:
            from rapidocr import RapidOCR

            ocr_engine = RapidOCR()
        self._ocr = ocr_engine

    def extract_ocr_stats(self, pil_img: Image.Image) -> tuple[int, int, str]:
        """Run RapidOCR on a downscaled copy and return (box_count, char_count, full_text)."""
        w, h = pil_img.size
        if w > 1200:
            img_small = pil_img.resize((1200, max(1, int(h * 1200 / w))), Image.Resampling.BILINEAR)
        else:
            img_small = pil_img

        res = self._ocr(np.asarray(img_small.convert("RGB")))
        txts = getattr(res, "txts", None)
        if txts:
            lines = [str(t).strip() for t in txts if str(t).strip()]
        elif isinstance(res, tuple) and res and res[0]:
            lines = [str(item[1]).strip() for item in res[0] if len(item) >= 2 and str(item[1]).strip()]
        else:
            lines = []

        full_text = " ".join(lines)
        return len(lines), len(full_text.replace(" ", "")), full_text

    def screen_pii(self, pil_img: Image.Image) -> dict[str, Any]:
        """Run OCR PII check, header Saraban type check, and visual quality tag extraction."""
        rgb = pil_img.convert("RGB")
        _, _, full_text = self.extract_ocr_stats(rgb)
        return {
            "has_pii": has_thai_citizen_id(full_text),
            "ocr_text": full_text,
            "detected_saraban_type": detect_header_saraban_type(full_text),
            "quality_tags": detect_quality_tags(rgb),
        }

    def screen_document(
        self,
        pil_img: Image.Image,
        min_dim: int = 800,
        max_saturation: float = 45.0,
        min_boxes: int = 8,
        min_chars: int = 60,
        max_aspect_ratio: float = 1.25,
    ) -> dict[str, Any]:
        """Evaluate whether a crawled image is an OCR-usable government document."""
        rgb = pil_img.convert("RGB")
        w, h = rgb.size
        min_dim_val = min(w, h)
        aspect_ratio = round(w / max(1, h), 3)

        hsv = rgb.convert("HSV")
        sat_channel = hsv.split()[1]
        mean_sat = round(ImageStat.Stat(sat_channel).mean[0], 2)

        gray = np.asarray(rgb.convert("L"))
        light_ratio = round(float(np.mean(gray >= 180)), 3)

        box_count, char_count, full_text = self.extract_ocr_stats(rgb)
        has_gov_marker = any(m in full_text for m in GOV_MARKERS)
        has_pii = has_thai_citizen_id(full_text)

        reasons: list[str] = []
        if min_dim_val < min_dim:
            reasons.append(f"low_res({w}x{h})")
        if mean_sat > max_saturation or (light_ratio < 0.40 and mean_sat > 25):
            reasons.append(f"colorful_graphic(sat={mean_sat},light={light_ratio})")
        if box_count < min_boxes or char_count < min_chars:
            reasons.append(f"low_text_density(boxes={box_count},chars={char_count})")
        if aspect_ratio > max_aspect_ratio and not has_gov_marker:
            reasons.append(f"wide_non_doc(ar={aspect_ratio})")

        return {
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "mean_saturation": mean_sat,
            "light_ratio": light_ratio,
            "ocr_boxes": box_count,
            "ocr_chars": char_count,
            "has_gov_marker": has_gov_marker,
            "has_pii": has_pii,
            "status": "rejected" if reasons else "kept",
            "reasons": reasons,
        }
