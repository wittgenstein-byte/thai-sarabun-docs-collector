"""Tests for the FastAPI ocr-screener microservice, OCRScreener, and ScreenerClient."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from train_ocr.screener import api as screener_api
from train_ocr.screener.client import ScreenerClient
from train_ocr.screener.engine import OCRScreener


class _StubRapidOCR:
    """Lightweight deterministic stub matching RapidOCR output shape."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __call__(self, _arr):
        return type("Result", (), {"txts": self._lines})()


def _png_bytes(width: int = 900, height: int = 1200, color: tuple[int, int, int] = (250, 250, 250)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def stub_api_client():
    sample_lines = [
        "ที่ ศธ 04025/4201",
        "สำนักงานเขตพื้นที่การศึกษา",
        "เรื่อง ขออนุญาตนำนักเรียนไปนอกสถานศึกษา",
        "เรียน ผู้อำนวยการโรงเรียนบ้านแดงน้อย",
        "ตามหนังสือที่อ้างถึง",
        "โรงเรียนขออนุญาตนำนักเรียนเข้าร่วมโครงการ",
        "ขอแสดงความนับถือ",
        "ลงชื่อ ผู้อำนวยการโรงเรียน พ.ศ. 2562",
    ]
    screener_api.set_screener(OCRScreener(ocr_engine=_StubRapidOCR(sample_lines)))
    try:
        yield TestClient(screener_api.app)
    finally:
        screener_api.set_screener(None)


def test_health_and_document_screening_api(stub_api_client: TestClient) -> None:
    resp = stub_api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "ocr-screener"}

    ok_resp = stub_api_client.post(
        "/screen/document",
        params={"min_dim": 800},
        files={"file": ("doc.png", _png_bytes(900, 1200), "image/png")},
    )
    assert ok_resp.status_code == 200
    payload = ok_resp.json()
    assert payload["status"] == "kept"
    assert payload["has_gov_marker"] is True
    assert payload["ocr_boxes"] == 8

    low_res_resp = stub_api_client.post(
        "/screen/document",
        params={"min_dim": 800},
        files={"file": ("small.png", _png_bytes(300, 400), "image/png")},
    )
    assert low_res_resp.status_code == 200
    assert low_res_resp.json()["status"] == "rejected"
    assert any("low_res" in r for r in low_res_resp.json()["reasons"])

    bad_resp = stub_api_client.post(
        "/screen/document",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert bad_resp.status_code == 400


def test_pii_screening_api() -> None:
    screener_api.set_screener(
        OCRScreener(ocr_engine=_StubRapidOCR(["บันทึกข้อความ", "เลขบัตร 1100702345676"]))
    )
    try:
        client = TestClient(screener_api.app)
        resp = client.post(
            "/screen/pii",
            files={"file": ("page.png", _png_bytes(800, 1000), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_pii"] is True
        assert data["detected_saraban_type"] == "internal_memo"
    finally:
        screener_api.set_screener(None)


def test_screener_client_local_and_remote_modes(monkeypatch: pytest.MonkeyPatch, stub_api_client: TestClient) -> None:
    # Remote mode via mocked requests.post forwarding to FastAPI TestClient
    def fake_post(url: str, files=None, params=None, timeout=60):
        path = "/" + url.split("://", 1)[1].split("/", 1)[1]
        return stub_api_client.post(path, files=files, params=params)

    monkeypatch.setattr("train_ocr.screener.client.requests.post", fake_post)
    remote = ScreenerClient(base_url="http://ocr-screener:8001")
    assert remote.is_remote is True
    res = remote.screen_document(Image.new("RGB", (900, 1200), (255, 255, 255)))
    assert res["status"] == "kept"

    # Local fallback mode (using injected stub on local client)
    local = ScreenerClient(base_url="")
    local._local_screener = OCRScreener(ocr_engine=_StubRapidOCR(["โลโก้"]))
    res_local = local.screen_document(Image.new("RGB", (240, 240), (255, 255, 255)))
    assert res_local["status"] == "rejected"


def test_real_rapidocr_on_pilot_samples_if_present() -> None:
    kept_sample = Path("data/google_crawl_test/download/เอกสารราชการ โรงเรียน/google_0001.jpg")
    logo_sample = Path("data/google_crawl_test/download/เอกสารราชการ โรงเรียน/google_0017.png")
    if not kept_sample.exists() or not logo_sample.exists():
        pytest.skip("Pilot crawl samples not present")

    screener = OCRScreener()
    with Image.open(kept_sample) as img_doc:
        res_doc = screener.screen_document(img_doc, min_dim=800)
    assert res_doc["status"] == "kept"
    assert res_doc["ocr_boxes"] >= 8
    assert res_doc["ocr_chars"] >= 60

    with Image.open(logo_sample) as img_logo:
        res_logo = screener.screen_document(img_logo, min_dim=800)
    assert res_logo["status"] == "rejected"


def test_filter_directory_preserves_subfolders_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from train_ocr.screener import filter_images

    class _FakeClient:
        def __init__(self, base_url=None):
            self.calls = 0

        def screen_document(self, _img, min_dim=800, min_boxes=8, min_chars=60):
            self.calls += 1
            return {
                "width": 1000,
                "height": 1400,
                "aspect_ratio": 0.714,
                "mean_saturation": 5.0,
                "light_ratio": 0.9,
                "ocr_boxes": 20,
                "ocr_chars": 200,
                "has_gov_marker": True,
                "has_pii": False,
                "status": "kept",
                "reasons": [],
            }

    fake = _FakeClient()
    monkeypatch.setattr(filter_images, "ScreenerClient", lambda base_url=None: fake)

    in_dir = tmp_path / "raw"
    kw1 = in_dir / "kw_obt"
    kw2 = in_dir / "kw_obj"
    kw1.mkdir(parents=True)
    kw2.mkdir(parents=True)

    img1 = Image.new("RGB", (1000, 1400), (250, 250, 250))
    img2 = Image.new("RGB", (1000, 1400), (245, 245, 245))
    img1.save(kw1 / "google_0000.png")
    # Byte-identical duplicate across keyword folders under the same filename (kw_obj sorts before kw_obt)
    (kw2 / "google_0000.png").write_bytes((kw1 / "google_0000.png").read_bytes())
    # Unique second image in kw_obt so both keyword subfolders appear under kept/
    img2.save(kw1 / "google_0001.png")

    out_dir = tmp_path / "filtered"
    records = filter_images.filter_directory(in_dir, out_dir, min_dim=800)

    assert len(records) == 3
    assert fake.calls == 2  # Duplicate skipped before OCR
    assert (out_dir / "kept" / "kw_obj" / "google_0000.png").exists()
    assert (out_dir / "kept" / "kw_obt" / "google_0001.png").exists()
    assert (out_dir / "rejected" / "kw_obt" / "google_0000.png").exists()
    dup_rec = [r for r in records if r["relative_path"] == "kw_obt/google_0000.png" or r["relative_path"] == "kw_obj/google_0000.png"]
    assert any(
        r["status"] == "rejected" and any(reason.startswith("duplicate(") for reason in r["reasons"])
        for r in dup_rec
    )


def test_screen_pdf_page_graphic_filters_posters_and_slides() -> None:
    sample_lines = [
        "ที่ มท 0808.2/ว 1234",
        "กรมส่งเสริมการปกครองท้องถิ่น",
        "เรื่อง แนวทางปฏิบัติ",
        "เรียน ผู้ว่าราชการจังหวัด",
        "ตามหนังสือที่อ้างถึง",
        "ขอแสดงความนับถือ ลงชื่อ",
    ]
    screener = OCRScreener(ocr_engine=_StubRapidOCR(sample_lines))

    # 1. White document page with small 3% red stamp -> kept
    doc_img = Image.new("RGB", (900, 1200), (250, 250, 250))
    arr = np.array(doc_img)
    arr[50:150, 50:320] = [220, 20, 20]  # ~2.5% red stamp
    res_doc = screener.screen_pdf_page_graphic(Image.fromarray(arr), title="แผนพัฒนาท้องถิ่น", embedded_chars=0)
    assert res_doc["status"] == "kept", res_doc

    # 2. Colorful poster / dark cover -> rejected
    poster_img = Image.new("RGB", (900, 1200), (20, 140, 220))
    res_poster = screener.screen_pdf_page_graphic(poster_img, title="รายงานประจำปี", embedded_chars=200)
    assert res_poster["status"] == "rejected"
    assert any("high_vivid_color" in r for r in res_poster["reasons"])
    assert any("non_paper_bg" in r for r in res_poster["reasons"])

    # 3. 16:9 slide -> rejected
    slide_img = Image.new("RGB", (1600, 900), (250, 250, 250))
    res_slide = screener.screen_pdf_page_graphic(slide_img, title="นำเสนอ", embedded_chars=200)
    assert res_slide["status"] == "rejected"
    assert any("slide_or_banner_ar" in r for r in res_slide["reasons"])

    # 4. Poster keyword in title -> rejected
    res_kw = screener.screen_pdf_page_graphic(doc_img, title="วิชาการ Weekly ฉบับที่ 10", embedded_chars=200)
    assert res_kw["status"] == "rejected"
    assert "poster_keyword" in res_kw["reasons"]
