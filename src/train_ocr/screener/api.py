"""FastAPI microservice exposing RapidOCR PII & document screening endpoints."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from train_ocr.screener.engine import OCRScreener

app = FastAPI(title="train-ocr Screener Service", version="0.2.0")

_screener: OCRScreener | None = None


def get_screener() -> OCRScreener:
    """Lazily initialize and cache the OCRScreener singleton."""
    global _screener
    if _screener is None:
        _screener = OCRScreener()
    return _screener


def set_screener(screener: OCRScreener | None) -> None:
    """Override the singleton screener (primarily for unit tests)."""
    global _screener
    _screener = screener


def _decode_upload(raw: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(raw)) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ocr-screener"}


@app.post("/screen/pii")
async def screen_pii(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    img = _decode_upload(raw)
    return get_screener().screen_pii(img)


@app.post("/screen/document")
async def screen_document(
    file: UploadFile = File(...),
    min_dim: int = Query(800, ge=1),
    min_boxes: int = Query(8, ge=0),
    min_chars: int = Query(60, ge=0),
) -> dict[str, Any]:
    raw = await file.read()
    img = _decode_upload(raw)
    return get_screener().screen_document(
        img,
        min_dim=min_dim,
        min_boxes=min_boxes,
        min_chars=min_chars,
    )
