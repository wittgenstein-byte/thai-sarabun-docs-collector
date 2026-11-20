"""Client abstraction for OCR screening (HTTP microservice or local in-process fallback)."""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import requests
from PIL import Image


class ScreenerClient:
    """Calls `ocr-screener` HTTP service when `OCR_SCREENER_URL` is set, else local OCRScreener."""

    def __init__(self, base_url: str | None = None, timeout: int = 60) -> None:
        resolved = base_url if base_url is not None else os.environ.get("OCR_SCREENER_URL", "")
        self.base_url = resolved.strip().rstrip("/")
        self.timeout = timeout
        self._local_screener: Any = None

    @property
    def is_remote(self) -> bool:
        return bool(self.base_url)

    def _get_local(self) -> Any:
        if self._local_screener is None:
            from train_ocr.screener.engine import OCRScreener

            self._local_screener = OCRScreener()
        return self._local_screener

    @staticmethod
    def _encode_png(pil_img: Image.Image) -> bytes:
        buf = BytesIO()
        pil_img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    def screen_pii(self, pil_img: Image.Image) -> dict[str, Any]:
        if self.base_url:
            png_bytes = self._encode_png(pil_img)
            resp = requests.post(
                f"{self.base_url}/screen/pii",
                files={"file": ("page.png", png_bytes, "image/png")},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        return self._get_local().screen_pii(pil_img)

    def screen_document(
        self,
        pil_img: Image.Image,
        min_dim: int = 800,
        min_boxes: int = 8,
        min_chars: int = 60,
    ) -> dict[str, Any]:
        if self.base_url:
            png_bytes = self._encode_png(pil_img)
            resp = requests.post(
                f"{self.base_url}/screen/document",
                params={"min_dim": min_dim, "min_boxes": min_boxes, "min_chars": min_chars},
                files={"file": ("image.png", png_bytes, "image/png")},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()
        return self._get_local().screen_document(
            pil_img,
            min_dim=min_dim,
            min_boxes=min_boxes,
            min_chars=min_chars,
        )
