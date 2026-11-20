"""Unit tests for the DLA bulk document harvester."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from train_ocr.collectors.dla_bulk.catalog import DlaCatalog
from train_ocr.collectors.dla_bulk.downloader import run_downloader
from train_ocr.collectors.dla_bulk.portals import (
    DLA_BASE,
    decode_tis620_filename,
    parse_dwr_ebook_path,
    parse_table_rows_generic,
    parse_total_pages,
    parse_total_records,
)


def _make_sample_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _make_sample_pdf_bytes() -> bytes:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 200)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_decode_tis620_filename() -> None:
    url = (
        "https://www.dla.go.th/download/OfficialDocFile/20261119/"
        "C28AC171_%C1%B7_0821.2-%C76320.pdf?notDownload=Y"
    )
    fname, ext = decode_tis620_filename(url)
    assert ext == ".pdf"
    assert "มท" in fname
    assert "ว6320" in fname


def test_parse_pagination_and_dwr() -> None:
    html = "<span>หน้าที่ 1 จาก 348</span><span>แสดง 1 ถึง 100 จาก 34740 รายการ</span>"
    assert parse_total_pages(html) == 348
    assert parse_total_records(html) == 34740

    dwr = 'throw "allowScriptTagRemoting";\r\ns0="/download/EbookFile/20261112/test_book.pdf";'
    assert parse_dwr_ebook_path(dwr) == "/download/EbookFile/20261112/test_book.pdf"


def test_parse_table_rows_generic() -> None:
    html = """
    <table>
      <tr class="tr-hover">
        <td>1</td>
        <td>กพส.</td>
        <td>มท 0821.2/ว 6320</td>
        <td>
          เรื่อง แจ้งแนวทางปฏิบัติ
          <a href="/download/OfficialDocFile/20261119/%C1%B7_0821.pdf?notDownload=Y">[หนังสือราชการ]</a>
          <a href="/download/OfficialDocFile/20261119/attach.png">[สิ่งที่ส่งมาด้วย]</a>
        </td>
        <td>19/11/2569</td>
      </tr>
    </table>
    """
    items = parse_table_rows_generic(html, DLA_BASE, "official_doc", "all", f"{DLA_BASE}/land/officialDoc.do")
    assert len(items) == 1
    doc, files = items[0]
    assert doc["dept"] == "กพส."
    assert doc["doc_no"] == "มท 0821.2/ว 6320"
    assert doc["title"] == "เรื่อง แจ้งแนวทางปฏิบัติ"
    assert len(files) == 2
    assert files[0]["ext"] == ".pdf"
    assert files[1]["ext"] == ".png"


def test_catalog_checkpoint_dedup_and_export(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.sqlite"
    with DlaCatalog(db_path) as cat:
        assert cat.get_checkpoint("official_doc", "all") == (0, 0, False)
        cat.save_checkpoint("official_doc", "all", 100, 34740, completed=False)
        assert cat.get_checkpoint("official_doc", "all") == (100, 34740, False)

        doc = {
            "doc_key": "doc1",
            "source": "official_doc",
            "subkey": "all",
            "dept": "กพส.",
            "doc_no": "ว1",
            "doc_date": "20/11/2569",
            "title": "ทดสอบ",
            "page_url": "https://www.dla.go.th/land/officialDoc.do",
        }
        files = [
            {"url": "https://www.dla.go.th/download/a.pdf", "label": "PDF", "original_filename": "a.pdf", "ext": ".pdf"},
            {"url": "https://www.dla.go.th/download/b.xlsx", "label": "XLS", "original_filename": "b.xlsx", "ext": ".xlsx"},
        ]
        nd, nf = cat.upsert_document_and_files(doc, files)
        assert (nd, nf) == (1, 2)

        pending_pdf_img = cat.fetch_pending_files(limit=10, allowed_exts=(".pdf", ".png"))
        assert len(pending_pdf_img) == 1
        assert pending_pdf_img[0]["url"].endswith("a.pdf")

        cat.mark_downloaded(
            "https://www.dla.go.th/download/a.pdf",
            sha256="abcd" * 16,
            size_bytes=2048,
            total_pages=2,
            local_path="files/official_doc/ab/abcdabcdabcdabcd.pdf",
            ext=".pdf",
        )
        assert cat.get_total_downloaded_bytes() == 2048
        assert cat.has_sha256("abcd" * 16) == "files/official_doc/ab/abcdabcdabcdabcd.pdf"

        manifest_path = tmp_path / "manifest.jsonl"
        assert cat.export_manifest_jsonl(manifest_path) == 1
        assert "a.pdf" in manifest_path.read_text(encoding="utf-8")


def test_run_downloader_pdf_image_dedup_and_budget(tmp_path: Path) -> None:
    pdf_bytes = _make_sample_pdf_bytes()
    png_bytes = _make_sample_png_bytes()
    if len(png_bytes) < 512:
        png_bytes = png_bytes + b"\x00" * 0  # keep valid PNG, use min_bytes=50 below

    db_path = tmp_path / "catalog.sqlite"
    files_dir = tmp_path / "files"

    url_pdf = "https://www.dla.go.th/download/doc1.pdf"
    url_pdf_dup = "https://www.dla.go.th/download/doc1_copy.pdf"
    url_png = "https://www.dla.go.th/download/img1.png"

    with DlaCatalog(db_path) as cat:
        cat.upsert_document_and_files(
            {"doc_key": "d1", "source": "official_doc", "title": "t1"},
            [
                {"url": url_pdf, "label": "1", "original_filename": "doc1.pdf", "ext": ".pdf"},
                {"url": url_pdf_dup, "label": "2", "original_filename": "doc1_copy.pdf", "ext": ".pdf"},
                {"url": url_png, "label": "3", "original_filename": "img1.png", "ext": ".png"},
            ],
        )

        payload_map = {
            url_pdf: (pdf_bytes, "application/pdf"),
            url_pdf_dup: (pdf_bytes, "application/pdf"),
            url_png: (png_bytes, "image/png"),
        }

        def fake_get(url: str, **_: object) -> MagicMock:
            data, ct = payload_map[url]
            resp = MagicMock()
            resp.status_code = 200
            resp.headers = {"Content-Type": ct, "Content-Length": str(len(data))}
            resp.iter_content.return_value = [data]
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            return resp

        mock_session = MagicMock()
        mock_session.get.side_effect = fake_get

        with patch("train_ocr.collectors.dla_bulk.downloader.make_session", return_value=mock_session):
            stats = run_downloader(
                cat,
                files_dir,
                allowed_exts=(".pdf", ".png"),
                workers=1,
                delay=0.0,
                min_bytes=50,
                max_total_bytes=10 * 1024 * 1024,
            )

        assert stats["downloaded"] == 2
        assert stats["duplicate"] == 1
        assert stats["failed"] == 0
        cat_stats = cat.get_stats()
        assert cat_stats["by_status"]["downloaded"]["count"] == 2
        assert cat_stats["by_status"]["duplicate"]["count"] == 1
