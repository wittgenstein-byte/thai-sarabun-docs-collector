"""Unit tests for the law.go.th bulk PDF collector."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from train_ocr.collectors.lawgoth import (
    LawCatalog,
    download_lawgoth_pdfs,
    encode_law_url,
    extract_pdf_files,
    index_lawgoth_pages,
)


def _make_sample_pdf_bytes() -> bytes:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 200)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_encode_law_url_and_extract() -> None:
    law_row = {
        "law_id": "11665",
        "table_of_law_id": "9114",
        "law_name_th": "พระราชบัญญัติทดสอบ",
        "content_files": [
            {
                "type": "content",
                "file_name": "ก0043 พระราชบัญญัติ 2484.pdf",
                "file_path": "https://apig.law.go.th/dga-backoffice-api/dga-storage/uploads/pdfKTB/ก0043 พระราชบัญญัติ 2484.pdf",
            },
            {
                "type": "content",
                "file_name": "doc.docx",
                "file_path": "https://apig.law.go.th/dga-backoffice-api/dga-storage/uploads/doc.docx",
            },
        ],
    }
    pdfs = extract_pdf_files(law_row)
    assert len(pdfs) == 1
    assert "%E0%B8%810043%20" in pdfs[0]["url"]
    assert pdfs[0]["url"].endswith(".pdf")
    assert encode_law_url("https://apig.law.go.th/a%20b.pdf") == "https://apig.law.go.th/a%20b.pdf"


def test_index_lawgoth_pages_and_download(tmp_path: Path) -> None:
    db_path = tmp_path / "catalog.sqlite"
    files_dir = tmp_path / "files"
    pdf_bytes = _make_sample_pdf_bytes()

    page1_json = {
        "rows": [
            {
                "law_id": "1",
                "table_of_law_id": "101",
                "law_name_th": "พ.ร.บ. 1",
                "content_files": [
                    {
                        "type": "content",
                        "file_name": "law1.pdf",
                        "file_path": "https://apig.law.go.th/uploads/law1.pdf",
                    }
                ],
            }
        ],
        "lastPage": 2,
        "currPage": 1,
    }
    page2_json = {
        "rows": [
            {
                "law_id": "2",
                "table_of_law_id": "102",
                "law_name_th": "พ.ร.บ. 2",
                "content_files": [
                    {
                        "type": "content",
                        "file_name": "law2.pdf",
                        "file_path": "https://apig.law.go.th/uploads/law2.pdf",
                    }
                ],
            }
        ],
        "lastPage": 2,
        "currPage": 2,
    }

    mock_session = MagicMock()
    resp1 = MagicMock(json=MagicMock(return_value=page1_json))
    resp2 = MagicMock(json=MagicMock(return_value=page2_json))
    mock_session.post.side_effect = [resp1, resp2]

    with LawCatalog(db_path) as cat:
        nl, nf = index_lawgoth_pages(mock_session, cat, page_size=1, delay=0.0)
        assert (nl, nf) == (2, 2)
        assert cat.get_checkpoint("main") == (2, 2, True)

        def fake_get(url: str, **_: object) -> MagicMock:
            r = MagicMock()
            r.status_code = 200
            r.iter_content.return_value = [pdf_bytes]
            r.__enter__.return_value = r
            r.__exit__.return_value = False
            return r

        dl_sess = MagicMock()
        dl_sess.get.side_effect = fake_get
        with patch("train_ocr.collectors.lawgoth.make_law_session", return_value=dl_sess):
            dl_stats = download_lawgoth_pdfs(cat, files_dir, workers=1, delay=0.0, min_bytes=50)

        assert dl_stats["downloaded"] == 1
        assert dl_stats["duplicate"] == 1
        assert cat.export_manifest(tmp_path / "manifest.jsonl") == 1
