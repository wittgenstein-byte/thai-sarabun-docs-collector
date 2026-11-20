"""Polite collector for Thai Saraban 6-type PDFs from dla.go.th/land/officialDoc.do."""

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup

from collect_pdfs import has_scanned_pages
from config import (
    MAX_PDF_BYTES,
    MIN_PDF_BYTES,
    SARABAN_MANIFEST_PATH,
    SARABAN_RAW_DIR,
    SARABAN_TYPES,
)
from saraban_utils import is_pii_blocked_title

DLA_URL = "https://www.dla.go.th/land/officialDoc.do"
DLA_BASE = "https://www.dla.go.th"
REQUEST_DELAY = 1.5
TIMEOUT = 30

# Query specs per Saraban type: (saraban_type, saraban_subtype, gbook_type, gbook_name)
DLA_QUERY_SPECS = [
    ("external_letter", "external_circular", "1", "", 1.0),
    ("internal_memo", "memo", "2", "", 1.0),
    ("stamped_letter", "stamped_letter", "3", "", 1.0),
    ("order_regulation", "order", "", "คำสั่ง", 0.45),
    ("order_regulation", "regulation", "", "ระเบียบ", 0.40),
    ("order_regulation", "bylaw", "", "ข้อบังคับ", 0.15),
    ("public_relations", "announcement", "4", "", 0.40),
    ("public_relations", "announcement", "", "ประกาศ", 0.60),
    ("official_record", "meeting_minutes", "", "รายงานการประชุม", 0.65),
    ("official_record", "meeting_minutes", "", "สรุปข้อราชการ", 0.15),
    ("official_record", "certificate_template", "", "หนังสือรับรอง", 0.20),
]


def load_saraban_manifest(manifest_path: Path) -> tuple[set[str], set[str], dict[str, int], dict[str, int]]:
    """Return (seen_urls, seen_hashes, counts_by_saraban_type, counts_by_subtype)."""
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    type_counts: dict[str, int] = {t: 0 for t in SARABAN_TYPES}
    subtype_counts: dict[str, int] = {}

    if not manifest_path.exists():
        return seen_urls, seen_hashes, type_counts, subtype_counts

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        seen_urls.add(rec["source_url"])
        seen_hashes.add(rec["sha256"])
        st = rec["saraban_type"]
        sub = rec.get("saraban_subtype", "")
        type_counts[st] = type_counts.get(st, 0) + 1
        subtype_counts[sub] = subtype_counts.get(sub, 0) + 1

    return seen_urls, seen_hashes, type_counts, subtype_counts


def post_dla_page(
    session: requests.Session, gbook_type: str, gbook_name: str, position: int, page_size: int = 25
) -> str:
    """POST multipart/form-data query to DLA officialDoc.do with retry."""
    files = {
        "cmd": (None, "list"),
        "data_index": (None, ""),
        "searchCondition.departmentLv2Id": (None, ""),
        "searchCondition.gbookType": (None, gbook_type),
        "searchCondition.gbookNumber": (None, ""),
        "searchCondition.gbookName": (None, gbook_name),
        "searchCondition.dateStart": (None, ""),
        "searchCondition.dateEnd": (None, ""),
        "searchCondition.pageSize": (None, str(page_size)),
        "searchCondition.position": (None, str(position)),
    }
    for attempt in range(2):
        try:
            resp = session.post(DLA_URL, files=files, timeout=TIMEOUT)
            resp.raise_for_status()
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(2.5)
                continue
            print(f"  [warn] DLA request failed (type={gbook_type}, kw={gbook_name}, pos={position}): {exc}")
            return ""
    return ""


def parse_total_pages(html: str) -> int:
    """Extract total page count from pagination dropdown."""
    m = re.search(r"\d+\s*จาก\s*(\d+)", html)
    return int(m.group(1)) if m else 1


def pick_best_link(
    links: list[tuple[str, str]], saraban_type: str, subtype: str, subject: str
) -> tuple[str, str] | None:
    """Select the most appropriate PDF link (url, label) for the target Saraban type."""
    pdf_links = [(u, lbl) for u, lbl in links if ".pdf" in u.lower()]
    if not pdf_links:
        return None

    skip_attach = ("สิ่งที่ส่งมาด้วย", "เอกสารแนบ", "สิ่งที่ส่ง")

    if saraban_type in {"external_letter", "stamped_letter"}:
        for u, lbl in pdf_links:
            if not any(s in lbl for s in skip_attach):
                return u, lbl
        return pdf_links[0]

    if saraban_type == "internal_memo":
        for u, lbl in pdf_links:
            clean = lbl.lstrip("[ ").strip()
            if clean.startswith(("คำสั่ง", "ประกาศ")) or any(s in lbl for s in skip_attach):
                continue
            return u, lbl
        return None

    if saraban_type == "order_regulation":
        kw_map = {"order": "คำสั่ง", "regulation": "ระเบียบ", "bylaw": "ข้อบังคับ"}
        kw = kw_map.get(subtype, "")
        for u, lbl in pdf_links:
            if kw and kw in lbl and "หนังสือแจ้ง" not in lbl:
                return u, lbl
        if kw and kw in subject:
            for u, lbl in pdf_links:
                if any(s in lbl for s in skip_attach):
                    return u, lbl
        return None

    if saraban_type == "public_relations":
        for u, lbl in pdf_links:
            if "ประกาศ" in lbl and not any(s in lbl for s in skip_attach):
                return u, lbl
        if "ประกาศ" in subject:
            for u, lbl in pdf_links:
                if any(s in lbl for s in skip_attach):
                    return u, lbl
        for u, lbl in pdf_links:
            if not any(s in lbl for s in skip_attach):
                return u, lbl
        return pdf_links[0]

    if saraban_type == "official_record":
        if subtype == "meeting_minutes":
            for u, lbl in pdf_links:
                if "รายงานการประชุม" in lbl or "สรุปข้อราชการ" in lbl:
                    return u, lbl
            if "รายงานการประชุม" in subject or "สรุปข้อราชการ" in subject:
                for u, lbl in pdf_links:
                    if any(s in lbl for s in skip_attach):
                        return u, lbl
                return pdf_links[0]
        else:
            for u, lbl in pdf_links:
                if "หนังสือรับรอง" in lbl or "แบบ" in lbl:
                    return u, lbl
            if "หนังสือรับรอง" in subject:
                for u, lbl in pdf_links:
                    if any(s in lbl for s in skip_attach):
                        return u, lbl
        return None

    return pdf_links[0]


def parse_dla_rows(html: str) -> list[dict]:
    """Parse <tr class='tr-hover'> rows from DLA HTML into structured candidate dicts."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    for tr in soup.select("tr.tr-hover"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        dept = tds[1].get_text(strip=True)
        doc_no = tds[2].get_text(strip=True)
        doc_date = tds[4].get_text(strip=True)

        links: list[tuple[str, str]] = []
        for a in tds[3].find_all("a", href=True):
            href = a["href"]
            if "OfficialDocFile" in href:
                links.append((urljoin(DLA_BASE, href), a.get_text(strip=True)))
            a.decompose()

        subject = " ".join(tds[3].get_text(" ", strip=True).split())
        if not links:
            continue

        results.append(
            {
                "dept": dept,
                "meta_doc_no": doc_no,
                "meta_doc_date": doc_date,
                "meta_subject": subject,
                "links": links,
            }
        )
    return results


def download_dla_pdf(
    session: requests.Session, url: str, dest_dir: Path, scanned_only: bool
) -> tuple[bool, str, int, int, Path | None]:
    """Download, validate, and optionally screen for scanned pages."""
    try:
        with session.get(url, stream=True, allow_redirects=True, timeout=TIMEOUT) as resp:
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
                if scanned_only:
                    sc, sc_ocr = has_scanned_pages(doc)
                    if sc + sc_ocr == 0:
                        dest_path.unlink(missing_ok=True)
                        return False, "", 0, 0, None
            finally:
                doc.close()

            return True, sha256, total, num_pages, dest_path
    except Exception:
        return False, "", 0, 0, None


def collect_dla(
    per_type: int,
    target_types: list[str],
    scanned_only: bool,
    seed: int,
) -> None:
    rng = random.Random(seed)
    SARABAN_RAW_DIR.mkdir(parents=True, exist_ok=True)
    SARABAN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_urls, seen_hashes, type_counts, subtype_counts = load_saraban_manifest(SARABAN_MANIFEST_PATH)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; thai-saraban-ocr-benchmark/1.0)"})
    try:
        session.get(DLA_URL, timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"Failed initial DLA session GET: {exc}")
        return

    with SARABAN_MANIFEST_PATH.open("a", encoding="utf-8") as mf:
        for saraban_type, subtype, gbook_type, gbook_name, sub_share in DLA_QUERY_SPECS:
            if saraban_type not in target_types:
                continue
            sub_target = max(1, round(per_type * sub_share)) if sub_share < 1.0 else per_type
            if type_counts.get(saraban_type, 0) >= per_type or subtype_counts.get(subtype, 0) >= sub_target:
                print(f"[{saraban_type}/{subtype}] already at target ({subtype_counts.get(subtype, 0)}/{sub_target}, total {type_counts.get(saraban_type, 0)}/{per_type}), skipping.")
                continue

            print(f"[{saraban_type}/{subtype}] querying DLA (sub {subtype_counts.get(subtype, 0)}/{sub_target}, type {type_counts.get(saraban_type, 0)}/{per_type})...")
            time.sleep(REQUEST_DELAY)
            first_html = post_dla_page(session, gbook_type, gbook_name, position=0, page_size=25)
            if not first_html:
                continue

            total_pages = parse_total_pages(first_html)
            # Sample page 0 plus spread pages for temporal/department diversity
            page_indices = [0]
            if total_pages > 1:
                others = list(range(1, min(total_pages, 120)))
                rng.shuffle(others)
                page_indices.extend(others[:24])

            for p_idx in page_indices:
                if type_counts.get(saraban_type, 0) >= per_type or subtype_counts.get(subtype, 0) >= sub_target:
                    break

                html = first_html if p_idx == 0 else post_dla_page(
                    session, gbook_type, gbook_name, position=p_idx * 25, page_size=25
                )
                if p_idx != 0:
                    time.sleep(REQUEST_DELAY)
                if not html:
                    continue

                rows = parse_dla_rows(html)
                rng.shuffle(rows)

                for row in rows:
                    if type_counts.get(saraban_type, 0) >= per_type or subtype_counts.get(subtype, 0) >= sub_target:
                        break
                    picked = pick_best_link(row["links"], saraban_type, subtype, row["meta_subject"])
                    if not picked:
                        continue
                    pdf_url, link_label = picked
                    if pdf_url in seen_urls:
                        continue
                    if is_pii_blocked_title(f"{row['meta_subject']} {link_label}"):
                        continue

                    time.sleep(REQUEST_DELAY)
                    ok, sha256, size_bytes, num_pages, dest_path = download_dla_pdf(
                        session, pdf_url, SARABAN_RAW_DIR / saraban_type, scanned_only=scanned_only
                    )
                    if not ok or dest_path is None:
                        continue
                    if sha256 in seen_hashes:
                        dest_path.unlink(missing_ok=True)
                        continue

                    record = {
                        "doc_id": sha256[:16],
                        "resource_id": sha256[:24],
                        "saraban_type": saraban_type,
                        "saraban_subtype": subtype,
                        "meta_doc_no": row["meta_doc_no"],
                        "meta_doc_date": row["meta_doc_date"],
                        "meta_subject": row["meta_subject"],
                        "link_label": link_label,
                        "organization": f"กรมส่งเสริมการปกครองท้องถิ่น ({row['dept']})" if row["dept"] else "กรมส่งเสริมการปกครองท้องถิ่น",
                        "source_portal": "dla.go.th",
                        "source_url": pdf_url,
                        "local_path": str(dest_path.as_posix()),
                        "sha256": sha256,
                        "size_bytes": size_bytes,
                        "total_pages": num_pages,
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    mf.flush()

                    seen_urls.add(pdf_url)
                    seen_hashes.add(sha256)
                    type_counts[saraban_type] = type_counts.get(saraban_type, 0) + 1
                    subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
                    print(
                        f"  + [{saraban_type} {type_counts[saraban_type]}/{per_type}] "
                        f"{subtype} | {row['meta_doc_no']} | {num_pages}p | {size_bytes // 1024}KB"
                    )

    print("\nDLA Collection Summary:", type_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect 6-type Thai Saraban PDFs from dla.go.th.")
    parser.add_argument("--per-type", type=int, default=45)
    parser.add_argument("--types", type=str, default=",".join(SARABAN_TYPES))
    parser.add_argument("--scanned-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target_types = [t.strip() for t in args.types.split(",") if t.strip()]
    collect_dla(
        per_type=args.per_type,
        target_types=target_types,
        scanned_only=args.scanned_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
