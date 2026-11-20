"""Supplementary collector pulling cross-agency Saraban PDFs from data.go.th."""

import argparse
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium
import requests

from collect_dla import load_saraban_manifest
from config import (
    ALLOWED_LICENSES,
    BASE_URL,
    MAX_PDF_BYTES,
    MIN_PDF_BYTES,
    REQUEST_TIMEOUT,
    SARABAN_MANIFEST_PATH,
    SARABAN_RAW_DIR,
    load_credentials,
)
from saraban_utils import is_pii_blocked_title

DATA_GOTH_QUERIES = [
    ("order_regulation", "order", "คำสั่ง"),
    ("order_regulation", "regulation", "ระเบียบ"),
    ("order_regulation", "bylaw", "ข้อบังคับ"),
    ("public_relations", "announcement", "ประกาศ"),
    ("public_relations", "press_release", "ข่าวประชาสัมพันธ์"),
    ("official_record", "meeting_minutes", "รายงานการประชุม"),
    ("official_record", "certificate_template", "หนังสือรับรอง"),
]


def download_valid_pdf(session: requests.Session, url: str, dest_dir: Path) -> tuple[bool, str, int, int, Path | None]:
    """Download and validate PDF from data.go.th resource URL."""
    try:
        with session.get(url, stream=True, allow_redirects=True, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status_code != 200:
                return False, "", 0, 0, None
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    return False, "", 0, 0, None
                chunks.append(chunk)
            if total < MIN_PDF_BYTES:
                return False, "", 0, 0, None

            raw = b"".join(chunks)
            if not raw.lstrip().startswith(b"%PDF-"):
                return False, "", 0, 0, None

            sha256 = hashlib.sha256(raw).hexdigest()
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{sha256[:16]}.pdf"
            dest_path.write_bytes(raw)

            doc = pdfium.PdfDocument(str(dest_path))
            try:
                num_pages = len(doc)
                if num_pages < 1:
                    dest_path.unlink(missing_ok=True)
                    return False, "", 0, 0, None
            finally:
                doc.close()

            return True, sha256, total, num_pages, dest_path
    except Exception:
        return False, "", 0, 0, None


def collect_extra(per_query: int = 10) -> None:
    creds = load_credentials()
    SARABAN_RAW_DIR.mkdir(parents=True, exist_ok=True)
    SARABAN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_urls, seen_hashes, type_counts, _ = load_saraban_manifest(SARABAN_MANIFEST_PATH)

    session = requests.Session()
    session.headers.update({"User-Agent": "thai-saraban-ocr-benchmark/1.0"})

    added = 0
    with SARABAN_MANIFEST_PATH.open("a", encoding="utf-8") as mf:
        for saraban_type, subtype, kw in DATA_GOTH_QUERIES:
            try:
                resp = session.get(
                    f"{BASE_URL}/package_search",
                    headers={"api-key": creds.datagoth_api_key},
                    params={"fq": "res_format:PDF", "q": kw, "rows": 80},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                pkgs = resp.json().get("result", {}).get("results", [])
            except Exception as exc:
                print(f"  [warn] data.go.th query failed for {kw}: {exc}")
                continue

            q_count = 0
            for pkg in pkgs:
                if q_count >= per_query:
                    break
                if (pkg.get("license_id") or "") not in ALLOWED_LICENSES:
                    continue
                title = pkg.get("title") or ""
                if is_pii_blocked_title(title):
                    continue

                org = (pkg.get("organization") or {}).get("title") or "data.go.th"
                for res in pkg.get("resources", []):
                    if (res.get("format") or "").strip().upper() != "PDF":
                        continue
                    url = res.get("url") or ""
                    if not url or url in seen_urls:
                        continue
                    res_name = res.get("name") or ""
                    if is_pii_blocked_title(res_name):
                        continue

                    ok, sha256, size_bytes, num_pages, dest_path = download_valid_pdf(
                        session, url, SARABAN_RAW_DIR / saraban_type
                    )
                    if not ok or dest_path is None:
                        continue
                    if sha256 in seen_hashes:
                        dest_path.unlink(missing_ok=True)
                        continue

                    record = {
                        "doc_id": pkg["id"],
                        "resource_id": res["id"],
                        "saraban_type": saraban_type,
                        "saraban_subtype": subtype,
                        "meta_doc_no": "",
                        "meta_doc_date": (pkg.get("metadata_modified") or "")[:10],
                        "meta_subject": title,
                        "link_label": res_name,
                        "organization": org,
                        "source_portal": "data.go.th",
                        "source_url": url,
                        "local_path": str(dest_path.as_posix()),
                        "sha256": sha256,
                        "size_bytes": size_bytes,
                        "total_pages": num_pages,
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    mf.flush()

                    seen_urls.add(url)
                    seen_hashes.add(sha256)
                    type_counts[saraban_type] = type_counts.get(saraban_type, 0) + 1
                    added += 1
                    q_count += 1
                    print(f"  + [data.go.th/{saraban_type}/{subtype}] {org[:30]} | {num_pages}p | {title[:40]}")
                    break

    print(f"\nExtra data.go.th collection added {added} PDFs. Updated totals: {type_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect supplementary Saraban PDFs from data.go.th.")
    parser.add_argument("--per-query", type=int, default=10)
    args = parser.parse_args()
    collect_extra(per_query=args.per_query)


if __name__ == "__main__":
    main()
