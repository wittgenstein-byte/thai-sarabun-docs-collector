"""Portal crawlers and HTML/DWR parsers for www.dla.go.th and sll.dla.go.th."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote_to_bytes, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from train_ocr.collectors.dla_bulk.catalog import DlaCatalog
from train_ocr.common.saraban_rules import is_pii_blocked_title

DLA_BASE = "https://www.dla.go.th"
SLL_BASE = "https://sll.dla.go.th"

ALL_SOURCES = (
    "official_doc",
    "legal_item",
    "ebook",
    "sll",
    "articles",
    "lgfis",
    "gov_forms",
    "int_articles",
    "ita",
    "emblem",
    "index_direct",
    "announce_news",
)

_SLL_CATEGORIES = ("2", "4", "27", "11", "13", "28", "31", "32")
_ITA_YEARS = ("2569", "2568", "2567")


def make_session(pool_maxsize: int = 16) -> requests.Session:
    """Create a requests.Session with connection pooling and polite retry."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "HEAD"),
    )
    adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; train-ocr-dla-bulk/1.0)",
            "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        }
    )
    return session


def decode_tis620_filename(raw_url: str) -> tuple[str, str]:
    """Decode percent-encoded TIS-620/CP874 or UTF-8 filename from a DLA download URL.

    Returns (decoded_filename, lowercase_ext_with_dot).
    Never mutates the URL itself; use the raw URL for HTTP requests.
    """
    parsed = urlparse(raw_url)
    raw_name = parsed.path.rsplit("/", 1)[-1]
    if not raw_name:
        return "", ""

    raw_bytes = unquote_to_bytes(raw_name)
    decoded = ""
    for enc in ("cp874", "utf-8", "latin-1"):
        try:
            decoded = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    ext = Path(decoded).suffix.lower() if "." in decoded else ""
    if len(ext) > 10:
        ext = ""
    return decoded, ext


def make_doc_key(source: str, subkey: str, doc_no: str, title: str, first_url: str) -> str:
    """Generate a deterministic, collision-resistant document key."""
    seed = f"{source}|{subkey}|{doc_no}|{title[:120]}|{first_url}"
    return hashlib.sha1(seed.encode("utf-8", errors="replace")).hexdigest()[:24]


def extract_download_links(container: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract all static /download/ attachment links inside an HTML element."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in container.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:", "#")):
            continue
        if "/download/" not in href and "OfficialDocFile" not in href:
            continue
        full_url = urljoin(base_url, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        label = " ".join(a.get_text(" ", strip=True).split())
        links.append((full_url, label))
    return links


def build_file_entries(links: list[tuple[str, str]]) -> list[dict]:
    """Convert (raw_url, label) pairs into catalog file entry dicts."""
    entries: list[dict] = []
    for url, label in links:
        fname, ext = decode_tis620_filename(url)
        entries.append(
            {
                "url": url,
                "label": label,
                "original_filename": fname,
                "ext": ext,
            }
        )
    return entries


def parse_total_pages(html: str) -> int:
    """Parse total page count from Struts pagination indicator (e.g. '1 จาก 348')."""
    m = re.search(r"\d+\s*จาก\s*(\d[\d,]*)", html)
    if m:
        return max(1, int(m.group(1).replace(",", "")))
    return 1


def parse_total_records(html: str) -> int:
    """Parse total record count from Struts pagination text (e.g. 'ถึง 100 จาก 34740')."""
    m = re.search(r"ถึง\s*[\d,]+\s*จาก\s*([\d,]+)", html)
    if m:
        return int(m.group(1).replace(",", ""))
    return 0


def post_struts_list(
    session: requests.Session,
    url: str,
    fields: dict[str, str],
    timeout: int = 30,
) -> str:
    """Submit a Struts multipart/form-data POST request and return UTF-8 HTML."""
    files = {k: (None, str(v)) for k, v in fields.items()}
    resp = session.post(url, files=files, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def get_html(session: requests.Session, url: str, timeout: int = 30) -> str:
    """GET an HTML page and decode as UTF-8."""
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def parse_table_rows_generic(
    html: str,
    base_url: str,
    source: str,
    subkey: str,
    page_url: str,
) -> list[tuple[dict, list[dict]]]:
    """Parse standard DLA <tr class='tr-hover'> or table rows containing /download/ links."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.tr-hover")
    if not rows:
        rows = [
            tr
            for tr in soup.find_all("tr")
            if tr.find("a", href=lambda h: bool(h and "/download/" in h))
        ]

    parsed: list[tuple[dict, list[dict]]] = []
    for tr in rows:
        links = extract_download_links(tr, base_url)
        if not links:
            continue

        tds = tr.find_all("td")
        dept = ""
        doc_no = ""
        doc_date = ""
        title = ""

        if len(tds) >= 5:
            dept = " ".join(tds[1].get_text(" ", strip=True).split())
            doc_no = " ".join(tds[2].get_text(" ", strip=True).split())
            doc_date = " ".join(tds[4].get_text(" ", strip=True).split())
            title_cell = BeautifulSoup(str(tds[3]), "html.parser")
            for a in title_cell.find_all("a"):
                a.decompose()
            title = " ".join(title_cell.get_text(" ", strip=True).split())
        else:
            texts = [" ".join(td.get_text(" ", strip=True).split()) for td in tds]
            title = " | ".join(t for t in texts if t)[:300]

        if not title:
            title = links[0][1] or decode_tis620_filename(links[0][0])[0]

        doc_key = make_doc_key(source, subkey, doc_no, title, links[0][0])
        doc = {
            "doc_key": doc_key,
            "source": source,
            "subkey": subkey,
            "dept": dept,
            "doc_no": doc_no,
            "doc_date": doc_date,
            "title": title,
            "page_url": page_url,
        }
        parsed.append((doc, build_file_entries(links)))
    return parsed


def _crawl_struts_paginated(
    *,
    session: requests.Session,
    catalog: DlaCatalog,
    source: str,
    subkey: str,
    url: str,
    base_fields: dict[str, str],
    page_size: int = 100,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    """Generic resumable crawler for Struts multipart/form-data paginated tables."""
    last_pos, _, completed = catalog.get_checkpoint(source, subkey)
    if completed:
        print(f"  [{source}:{subkey}] already completed, skipping.")
        return 0, 0

    try:
        session.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"  [warn] [{source}:{subkey}] initial GET failed: {exc}")

    pos = last_pos
    pages_crawled = 0
    total_new_docs = 0
    total_new_files = 0
    total_pages = 1

    while True:
        if max_pages is not None and pages_crawled >= max_pages:
            break

        fields = dict(base_fields)
        fields["searchCondition.pageSize"] = str(page_size)
        fields["searchCondition.position"] = str(pos)

        try:
            html = post_struts_list(session, url, fields)
        except requests.RequestException as exc:
            print(f"  [warn] [{source}:{subkey}] pos={pos} failed: {exc}")
            break

        total_pages = parse_total_pages(html)
        total_records = parse_total_records(html)
        items = parse_table_rows_generic(html, DLA_BASE, source, subkey, url)
        if not items:
            catalog.save_checkpoint(source, subkey, pos, total_records, completed=True)
            break

        page_new_files = 0
        for doc, files in items:
            if skip_pii and is_pii_blocked_title(doc["title"]):
                continue
            nd, nf = catalog.upsert_document_and_files(doc, files)
            total_new_docs += nd
            total_new_files += nf
            page_new_files += nf

        pages_crawled += 1
        next_pos = pos + page_size
        current_page = (pos // page_size) + 1
        is_done = current_page >= total_pages
        catalog.save_checkpoint(source, subkey, next_pos, total_records, completed=is_done)

        print(
            f"  [{source}:{subkey}] page {current_page}/{total_pages} "
            f"(pos={pos}, rows={len(items)}, +{page_new_files} files)"
        )

        if is_done:
            break
        pos = next_pos
        if delay > 0:
            time.sleep(delay)

    return total_new_docs, total_new_files


def crawl_official_doc(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    return _crawl_struts_paginated(
        session=session,
        catalog=catalog,
        source="official_doc",
        subkey="all",
        url=f"{DLA_BASE}/land/officialDoc.do",
        base_fields={
            "cmd": "list",
            "data_index": "",
            "searchCondition.departmentLv2Id": "",
            "searchCondition.gbookType": "",
            "searchCondition.gbookNumber": "",
            "searchCondition.gbookName": "",
            "searchCondition.dateStart": "",
            "searchCondition.dateEnd": "",
        },
        page_size=100,
        delay=delay,
        max_pages=max_pages,
        skip_pii=skip_pii,
    )


def crawl_announce_news(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    return _crawl_struts_paginated(
        session=session,
        catalog=catalog,
        source="announce_news",
        subkey="all",
        url=f"{DLA_BASE}/land/annouceNews.do",
        base_fields={
            "cmd": "list",
            "data_index": "",
            "searchCondition.departmentType": "",
            "searchCondition.newsEpicId": "",
            "searchCondition.newsTypeId": "",
            "searchCondition.newsCategoryId": "",
            "searchCondition.fromDate": "",
            "searchCondition.toDate": "",
            "searchCondition.newsName": "",
        },
        page_size=100,
        delay=delay,
        max_pages=max_pages,
        skip_pii=skip_pii,
    )


def crawl_legal_item(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    docs_total = 0
    files_total = 0

    for tab in ("0", "2"):
        nd, nf = _crawl_struts_paginated(
            session=session,
            catalog=catalog,
            source="legal_item",
            subkey=f"tab{tab}",
            url=f"{DLA_BASE}/land/iframeDoc.do?cmd=goList&tab={tab}",
            base_fields={
                "cmd": "list",
                "data_index": "",
                "searchCondition.departmentLv2Id": "",
                "searchCondition.gbookNumber": "",
                "searchCondition.gbookName": "",
                "searchCondition.dateStart": "",
                "searchCondition.dateEnd": "",
            },
            page_size=100,
            delay=delay,
            max_pages=max_pages,
            skip_pii=skip_pii,
        )
        docs_total += nd
        files_total += nf

    nd, nf = _crawl_struts_paginated(
        session=session,
        catalog=catalog,
        source="legal_item",
        subkey="tab3",
        url=f"{DLA_BASE}/land/legalItemFon.do",
        base_fields={
            "cmd": "goTab",
            "data_index": "3",
            "searchCondition.lawType": "",
            "searchCondition.gbookName": "",
        },
        page_size=100,
        delay=delay,
        max_pages=max_pages,
        skip_pii=skip_pii,
    )
    return docs_total + nd, files_total + nf


def crawl_articles(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    return _crawl_struts_paginated(
        session=session,
        catalog=catalog,
        source="articles",
        subkey="all",
        url=f"{DLA_BASE}/land/articles.do",
        base_fields={"cmd": "list", "data_index": ""},
        page_size=100,
        delay=delay,
        max_pages=max_pages,
        skip_pii=skip_pii,
    )


def crawl_int_articles(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    return _crawl_struts_paginated(
        session=session,
        catalog=catalog,
        source="int_articles",
        subkey="all",
        url=f"{DLA_BASE}/land/intarticlefon.do",
        base_fields={"cmd": "list", "data_index": ""},
        page_size=100,
        delay=delay,
        max_pages=max_pages,
        skip_pii=skip_pii,
    )


def crawl_lgfis(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    return _crawl_struts_paginated(
        session=session,
        catalog=catalog,
        source="lgfis",
        subkey="all",
        url=f"{DLA_BASE}/land/lgfisfon.do",
        base_fields={
            "cmd": "list",
            "data_index": "",
            "searchCondition.revenueType": "",
            "searchCondition.budgetYear": "",
        },
        page_size=100,
        delay=delay,
        max_pages=max_pages,
        skip_pii=skip_pii,
    )


def crawl_gov_forms(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    docs_total = 0
    files_total = 0
    for idx in range(10):
        subkey = f"group{idx}"
        _, _, completed = catalog.get_checkpoint("gov_forms", subkey)
        if completed:
            continue
        url = f"{DLA_BASE}/land/govloadfon.do?cmd=list&data_index={idx}"
        try:
            html = get_html(session, url)
        except requests.RequestException as exc:
            print(f"  [warn] [gov_forms:{subkey}] GET failed: {exc}")
            continue
        items = parse_table_rows_generic(html, DLA_BASE, "gov_forms", subkey, url)
        for doc, files in items:
            if skip_pii and is_pii_blocked_title(doc["title"]):
                continue
            nd, nf = catalog.upsert_document_and_files(doc, files)
            docs_total += nd
            files_total += nf
        catalog.save_checkpoint("gov_forms", subkey, len(items), len(items), completed=True)
        if delay > 0:
            time.sleep(delay)
        if max_pages is not None and idx + 1 >= max_pages:
            break
    print(f"  [gov_forms] indexed +{files_total} files")
    return docs_total, files_total


def crawl_ita(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    docs_total = 0
    files_total = 0
    url = f"{DLA_BASE}/land/itaFon.do"
    for i, yr in enumerate(_ITA_YEARS):
        subkey = f"year_{yr}"
        _, _, completed = catalog.get_checkpoint("ita", subkey)
        if completed:
            continue
        try:
            html = post_struts_list(
                session,
                url,
                {"cmd": "list", "data_index": "", "searchCondition.budgetYear": yr},
            )
        except requests.RequestException as exc:
            print(f"  [warn] [ita:{subkey}] POST failed: {exc}")
            continue
        items = parse_table_rows_generic(html, DLA_BASE, "ita", subkey, url)
        for doc, files in items:
            if skip_pii and is_pii_blocked_title(doc["title"]):
                continue
            nd, nf = catalog.upsert_document_and_files(doc, files)
            docs_total += nd
            files_total += nf
        catalog.save_checkpoint("ita", subkey, len(items), len(items), completed=True)
        if delay > 0:
            time.sleep(delay)
        if max_pages is not None and i + 1 >= max_pages:
            break
    print(f"  [ita] indexed +{files_total} files")
    return docs_total, files_total


def crawl_emblem(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    docs_total = 0
    files_total = 0
    for idx in ("1", "2", "3"):
        subkey = f"cat{idx}"
        _, _, completed = catalog.get_checkpoint("emblem", subkey)
        if completed:
            continue
        url = f"{DLA_BASE}/land/emblemTempFon.do?cmd=list&data_index={idx}"
        try:
            html = get_html(session, url)
        except requests.RequestException as exc:
            print(f"  [warn] [emblem:{subkey}] GET failed: {exc}")
            continue
        soup = BeautifulSoup(html, "html.parser")
        links = extract_download_links(soup, DLA_BASE)
        if links:
            doc_key = make_doc_key("emblem", subkey, "", f"emblem_{subkey}", links[0][0])
            doc = {
                "doc_key": doc_key,
                "source": "emblem",
                "subkey": subkey,
                "dept": "",
                "doc_no": "",
                "doc_date": "",
                "title": f"ตราสัญลักษณ์/แม่แบบ หมวด {idx}",
                "page_url": url,
            }
            nd, nf = catalog.upsert_document_and_files(doc, build_file_entries(links))
            docs_total += nd
            files_total += nf
        catalog.save_checkpoint("emblem", subkey, len(links), len(links), completed=True)
        if delay > 0:
            time.sleep(delay)
        if max_pages is not None and int(idx) >= max_pages:
            break
    print(f"  [emblem] indexed +{files_total} files")
    return docs_total, files_total


def crawl_index_direct(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    _, _, completed = catalog.get_checkpoint("index_direct", "home")
    if completed:
        return 0, 0
    url = f"{DLA_BASE}/index.do"
    try:
        html = get_html(session, url)
    except requests.RequestException as exc:
        print(f"  [warn] [index_direct] GET failed: {exc}")
        return 0, 0

    soup = BeautifulSoup(html, "html.parser")
    links = extract_download_links(soup, DLA_BASE)
    docs_total = 0
    files_total = 0
    for full_url, label in links:
        title = label or decode_tis620_filename(full_url)[0]
        if skip_pii and is_pii_blocked_title(title):
            continue
        doc = {
            "doc_key": make_doc_key("index_direct", "home", "", title, full_url),
            "source": "index_direct",
            "subkey": "home",
            "dept": "",
            "doc_no": "",
            "doc_date": "",
            "title": title,
            "page_url": url,
        }
        nd, nf = catalog.upsert_document_and_files(doc, build_file_entries([(full_url, label)]))
        docs_total += nd
        files_total += nf
    catalog.save_checkpoint("index_direct", "home", len(links), len(links), completed=True)
    print(f"  [index_direct] indexed +{files_total} files")
    return docs_total, files_total


def parse_dwr_ebook_path(dwr_text: str) -> str | None:
    """Extract the PDF/file path returned by DWR SsoService.getEbookFile.dwr."""
    m = re.search(r"['\"]((?:/download/|https?://)[^'\"]+)['\"]", dwr_text)
    if m:
        return m.group(1)
    m2 = re.search(r"s\d+\s*=\s*\"([^\"]+\.(?:pdf|PDF))\"", dwr_text)
    if m2:
        return m2.group(1)
    return None


def _resolve_ebook_files(session: requests.Session, ebook_token: str) -> list[tuple[str, str]]:
    """Resolve an e-book tokenId through goView -> form2 -> DWR getEbookFile."""
    goview_url = f"{DLA_BASE}/land/ebook.do?cmd=goView&tokenId={ebook_token}"
    html = get_html(session, goview_url)
    soup = BeautifulSoup(html, "html.parser")
    file_tokens = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"cmd=form2&tokenId=([^&'\"]+)", a["href"])
        if m:
            label = " ".join(a.get_text(" ", strip=True).split())
            file_tokens.append((m.group(1), label))

    resolved: list[tuple[str, str]] = []
    for ftok, label in file_tokens:
        try:
            get_html(session, f"{DLA_BASE}/land/ebook.do?cmd=form2&tokenId={ftok}")
            dwr_body = (
                "callCount=1\n"
                "page=/land/ebook.do\n"
                "httpSessionId=\n"
                "scriptSessionId=1\n"
                "c0-scriptName=SsoService\n"
                "c0-methodName=getEbookFile\n"
                "c0-id=0\n"
                f"c0-param0=string:{ftok}\n"
                "batchId=0\n"
            )
            resp = session.post(
                f"{DLA_BASE}/dwr/call/plaincall/SsoService.getEbookFile.dwr",
                data=dwr_body.encode("utf-8"),
                headers={"Content-Type": "text/plain"},
                timeout=30,
            )
            resp.raise_for_status()
            path = parse_dwr_ebook_path(resp.text)
            if path:
                resolved.append((urljoin(DLA_BASE, path), label))
        except requests.RequestException:
            continue
    return resolved


def crawl_ebook(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    last_pos, _, completed = catalog.get_checkpoint("ebook", "all")
    if completed:
        print("  [ebook:all] already completed, skipping.")
        return 0, 0

    url = f"{DLA_BASE}/land/ebook.do"
    try:
        session.get(url, timeout=30)
    except requests.RequestException:
        pass

    pos = last_pos
    page_size = 50
    pages_crawled = 0
    docs_total = 0
    files_total = 0

    while True:
        if max_pages is not None and pages_crawled >= max_pages:
            break
        try:
            html = post_struts_list(
                session,
                url,
                {
                    "cmd": "list",
                    "data_index": "",
                    "searchCondition.ebookYear": "",
                    "searchCondition.ebookType": "",
                    "searchCondition.departmentLv2Id": "",
                    "searchCondition.ebookName": "",
                    "searchCondition.pageSize": str(page_size),
                    "searchCondition.position": str(pos),
                },
            )
        except requests.RequestException as exc:
            print(f"  [warn] [ebook] pos={pos} failed: {exc}")
            break

        total_pages = parse_total_pages(html)
        total_records = parse_total_records(html)
        tokens = re.findall(r"addViewCountAndModal\(\s*['\"]([^'\"]+)['\"]", html)
        if not tokens:
            catalog.save_checkpoint("ebook", "all", pos, total_records, completed=True)
            break

        soup = BeautifulSoup(html, "html.parser")
        seen_tok: set[str] = set()
        unique_tokens = [t for t in tokens if not (t in seen_tok or seen_tok.add(t))]

        page_files = 0
        for tok in unique_tokens:
            title = f"ebook_{tok[:12]}"
            el = soup.find(attrs={"onclick": re.compile(re.escape(tok))})
            if el is not None:
                parent_txt = " ".join(el.parent.get_text(" ", strip=True).split())
                if parent_txt:
                    title = parent_txt[:240]
            if skip_pii and is_pii_blocked_title(title):
                continue

            links = _resolve_ebook_files(session, tok)
            if not links:
                continue
            doc = {
                "doc_key": make_doc_key("ebook", "all", tok, title, links[0][0]),
                "source": "ebook",
                "subkey": "all",
                "dept": "",
                "doc_no": tok,
                "doc_date": "",
                "title": title,
                "page_url": url,
            }
            nd, nf = catalog.upsert_document_and_files(doc, build_file_entries(links))
            docs_total += nd
            files_total += nf
            page_files += nf
            if delay > 0:
                time.sleep(min(delay, 0.3))

        pages_crawled += 1
        next_pos = pos + page_size
        current_page = (pos // page_size) + 1
        is_done = current_page >= total_pages
        catalog.save_checkpoint("ebook", "all", next_pos, total_records, completed=is_done)
        print(f"  [ebook] page {current_page}/{total_pages} (tokens={len(unique_tokens)}, +{page_files} files)")
        if is_done:
            break
        pos = next_pos

    return docs_total, files_total


def crawl_sll(
    session: requests.Session,
    catalog: DlaCatalog,
    *,
    delay: float = 0.5,
    max_pages: int | None = None,
    skip_pii: bool = False,
) -> tuple[int, int]:
    docs_total = 0
    files_total = 0
    try:
        session.get(f"{SLL_BASE}/public/searchSystem.do", timeout=30)
    except requests.RequestException:
        pass

    for i, cat_id in enumerate(_SLL_CATEGORIES):
        subkey = f"cat_{cat_id}"
        _, _, completed = catalog.get_checkpoint("sll", subkey)
        if completed:
            continue

        cat_url = f"{SLL_BASE}/public/searchSystemIframe.do?cmd=doCategoryStory&data_index={cat_id}"
        try:
            cat_html = get_html(session, cat_url)
        except requests.RequestException as exc:
            print(f"  [warn] [sll:{subkey}] GET failed: {exc}")
            continue

        sub_ids = sorted(
            set(re.findall(r"doSubCategory[^'\"]*?(\d+)", cat_html))
            | set(re.findall(r"data_index=(\d+)", cat_html))
        )
        cat_files = 0
        for sub_id in sub_ids:
            sub_url = f"{SLL_BASE}/public/searchSystemIframe.do?cmd=doSubCategory&data_index={sub_id}"
            try:
                sub_html = get_html(session, sub_url)
            except requests.RequestException:
                continue
            soup = BeautifulSoup(sub_html, "html.parser")
            links = extract_download_links(soup, SLL_BASE)
            for full_url, label in links:
                title = label or decode_tis620_filename(full_url)[0]
                if skip_pii and is_pii_blocked_title(title):
                    continue
                doc = {
                    "doc_key": make_doc_key("sll", f"{subkey}_{sub_id}", "", title, full_url),
                    "source": "sll",
                    "subkey": f"{subkey}_{sub_id}",
                    "dept": "",
                    "doc_no": "",
                    "doc_date": "",
                    "title": title,
                    "page_url": sub_url,
                }
                nd, nf = catalog.upsert_document_and_files(doc, build_file_entries([(full_url, label)]))
                docs_total += nd
                files_total += nf
                cat_files += nf
            if delay > 0:
                time.sleep(min(delay, 0.3))

        catalog.save_checkpoint("sll", subkey, len(sub_ids), len(sub_ids), completed=True)
        print(f"  [sll:{subkey}] subcats={len(sub_ids)}, +{cat_files} files")
        if max_pages is not None and i + 1 >= max_pages:
            break

    return docs_total, files_total


CRAWLERS: dict[str, Callable[..., tuple[int, int]]] = {
    "official_doc": crawl_official_doc,
    "announce_news": crawl_announce_news,
    "legal_item": crawl_legal_item,
    "ebook": crawl_ebook,
    "sll": crawl_sll,
    "articles": crawl_articles,
    "int_articles": crawl_int_articles,
    "lgfis": crawl_lgfis,
    "gov_forms": crawl_gov_forms,
    "ita": crawl_ita,
    "emblem": crawl_emblem,
    "index_direct": crawl_index_direct,
}
