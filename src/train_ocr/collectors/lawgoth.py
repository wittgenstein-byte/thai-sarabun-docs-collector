"""Resumable bulk PDF collector for law.go.th (apig.law.go.th)."""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from train_ocr.common.manifest import write_jsonl
from train_ocr.common.settings import LAWGOTH_DIR

LAWGOTH_API_BASE = "https://apig.law.go.th/dga-user-service-phase2"
LAWGOTH_LIST_URL = f"{LAWGOTH_API_BASE}/law"
LAWGOTH_DETAIL_URL = f"{LAWGOTH_API_BASE}/law/detail"
LAWGOTH_API_KEY = "4nEZYvTwRFlUVn7aK85cZ2xSU83dOFai"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def encode_law_url(raw_url: str) -> str:
    """Percent-encode Thai/space characters in a law.go.th file_path URL safely."""
    parts = urlsplit(raw_url.strip())
    encoded_path = quote(parts.path, safe="/%")
    encoded_query = quote(parts.query, safe="=&%")
    return urlunsplit((parts.scheme, parts.netloc, encoded_path, encoded_query, parts.fragment))


def make_law_session(pool_maxsize: int = 16) -> requests.Session:
    """Create a requests.Session with x-api-key and retry/backoff for apig.law.go.th."""
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; train-ocr-lawgoth/1.0)",
            "Accept": "application/json, application/pdf, */*",
            "x-api-key": LAWGOTH_API_KEY,
        }
    )
    return session


class LawCatalog:
    """Thread-safe SQLite WAL catalog for law.go.th pagination checkpoints & PDF downloads."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS crawl_state (
                key TEXT PRIMARY KEY,
                last_page INTEGER NOT NULL DEFAULT 0,
                total_pages INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS laws (
                law_key TEXT PRIMARY KEY,
                law_id TEXT NOT NULL DEFAULT '',
                table_of_law_id TEXT NOT NULL DEFAULT '',
                law_name_th TEXT NOT NULL DEFAULT '',
                annouce_date TEXT NOT NULL DEFAULT '',
                effective_startdate TEXT NOT NULL DEFAULT '',
                category_id TEXT NOT NULL DEFAULT '',
                agency_response TEXT NOT NULL DEFAULT '',
                hirachy_of_law_id TEXT NOT NULL DEFAULT '',
                discovered_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                url TEXT PRIMARY KEY,
                law_key TEXT NOT NULL,
                file_type TEXT NOT NULL DEFAULT '',
                file_name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                sha256 TEXT NOT NULL DEFAULT '',
                size_bytes INTEGER NOT NULL DEFAULT 0,
                total_pages INTEGER NOT NULL DEFAULT 0,
                local_path TEXT NOT NULL DEFAULT '',
                http_status INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                retries INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_law_files_status ON files(status);
            CREATE INDEX IF NOT EXISTS idx_law_files_sha256 ON files(sha256);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> LawCatalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_checkpoint(self, key: str = "main") -> tuple[int, int, bool]:
        with self._lock:
            row = self._conn.execute(
                "SELECT last_page, total_pages, completed FROM crawl_state WHERE key=?",
                (key,),
            ).fetchone()
            if row is None:
                return 0, 0, False
            return int(row["last_page"]), int(row["total_pages"]), bool(row["completed"])

    def save_checkpoint(self, key: str, last_page: int, total_pages: int, completed: bool) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO crawl_state (key, last_page, total_pages, completed, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    last_page=excluded.last_page,
                    total_pages=MAX(crawl_state.total_pages, excluded.total_pages),
                    completed=excluded.completed,
                    updated_at=excluded.updated_at
                """,
                (key, last_page, total_pages, int(completed), now),
            )
            self._conn.commit()

    def upsert_law_and_files(self, law_row: dict, pdf_files: list[dict]) -> tuple[int, int]:
        now = _utc_now()
        law_key = str(law_row.get("table_of_law_id") or law_row.get("law_id") or "")
        if not law_key:
            return 0, 0
        with self._lock:
            cur_law = self._conn.execute(
                """
                INSERT OR IGNORE INTO laws
                    (law_key, law_id, table_of_law_id, law_name_th, annouce_date,
                     effective_startdate, category_id, agency_response, hirachy_of_law_id, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    law_key,
                    str(law_row.get("law_id") or ""),
                    str(law_row.get("table_of_law_id") or ""),
                    str(law_row.get("law_name_th") or law_row.get("law_name_og") or ""),
                    str(law_row.get("annouce_date") or ""),
                    str(law_row.get("effective_startdate") or ""),
                    str(law_row.get("category_id") or ""),
                    str(law_row.get("agency_response") or ""),
                    str(law_row.get("hirachy_of_law_id") or ""),
                    now,
                ),
            )
            new_files = 0
            for f in pdf_files:
                cur_f = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO files
                        (url, law_key, file_type, file_name, status, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', ?)
                    """,
                    (f["url"], law_key, f.get("file_type", ""), f.get("file_name", ""), now),
                )
                new_files += cur_f.rowcount
            self._conn.commit()
            return cur_law.rowcount, new_files

    def fetch_pending_files(
        self,
        limit: int = 100,
        include_failed: bool = False,
        max_retries: int = 3,
    ) -> list[dict]:
        statuses = ["'pending'"]
        if include_failed:
            statuses.append("'failed'")
        sql = f"""
            SELECT url, law_key, file_type, file_name, retries
            FROM files
            WHERE status IN ({','.join(statuses)}) AND retries < ?
            ORDER BY retries ASC, updated_at ASC
            LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, (max_retries, limit)).fetchall()
            return [dict(r) for r in rows]

    def has_sha256(self, sha256: str) -> str | None:
        if not sha256:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT local_path FROM files WHERE sha256=? AND status='downloaded' AND local_path != '' LIMIT 1",
                (sha256,),
            ).fetchone()
            return str(row["local_path"]) if row else None

    def get_total_downloaded_bytes(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM files WHERE status='downloaded'"
            ).fetchone()
            return int(row["total"]) if row else 0

    def mark_downloaded(
        self,
        url: str,
        *,
        sha256: str,
        size_bytes: int,
        total_pages: int,
        local_path: str,
        http_status: int = 200,
    ) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE files
                SET status='downloaded', sha256=?, size_bytes=?, total_pages=?,
                    local_path=?, http_status=?, error='', updated_at=?
                WHERE url=?
                """,
                (sha256, size_bytes, total_pages, local_path, http_status, now, url),
            )
            self._conn.commit()

    def mark_duplicate(
        self,
        url: str,
        *,
        sha256: str,
        size_bytes: int,
        existing_path: str,
        http_status: int = 200,
    ) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE files
                SET status='duplicate', sha256=?, size_bytes=?, local_path=?,
                    http_status=?, error='', updated_at=?
                WHERE url=?
                """,
                (sha256, size_bytes, existing_path, http_status, now, url),
            )
            self._conn.commit()

    def mark_skipped(self, url: str, reason: str, http_status: int = 0) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                "UPDATE files SET status='skipped', error=?, http_status=?, updated_at=? WHERE url=?",
                (reason[:300], http_status, now, url),
            )
            self._conn.commit()

    def mark_failed(self, url: str, error: str, http_status: int = 0) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                "UPDATE files SET status='failed', error=?, http_status=?, retries=retries+1, updated_at=? WHERE url=?",
                (error[:300], http_status, now, url),
            )
            self._conn.commit()

    def export_manifest(self, manifest_path: Path) -> int:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    f.url AS source_url,
                    f.file_type,
                    f.file_name,
                    f.sha256,
                    f.size_bytes,
                    f.total_pages,
                    f.local_path,
                    l.law_id,
                    l.table_of_law_id,
                    l.law_name_th,
                    l.annouce_date,
                    l.effective_startdate,
                    l.category_id,
                    l.agency_response,
                    l.hirachy_of_law_id
                FROM files f
                JOIN laws l ON f.law_key = l.law_key
                WHERE f.status = 'downloaded'
                ORDER BY l.table_of_law_id, f.url
                """
            ).fetchall()

        records = []
        for r in rows:
            rec = dict(r)
            rec["doc_id"] = rec["sha256"][:16]
            rec["source_portal"] = "law.go.th"
            records.append(rec)
        write_jsonl(manifest_path, records)
        return len(records)

    def get_stats(self) -> dict:
        with self._lock:
            law_total = self._conn.execute("SELECT COUNT(*) FROM laws").fetchone()[0]
            by_status = {
                r["status"]: {"count": r["cnt"], "bytes": r["bytes"]}
                for r in self._conn.execute(
                    "SELECT status, COUNT(*) AS cnt, COALESCE(SUM(size_bytes), 0) AS bytes FROM files GROUP BY status"
                ).fetchall()
            }
        return {"laws_indexed": int(law_total), "by_status": by_status}


def extract_pdf_files(law_row: dict) -> list[dict]:
    """Extract downloadable PDF entries from a law row's content_files."""
    pdf_entries: list[dict] = []
    seen: set[str] = set()
    for cf in law_row.get("content_files") or []:
        raw_path = (cf.get("file_path") or "").strip()
        fname = (cf.get("file_name") or "").strip()
        if not raw_path:
            continue
        if not (raw_path.lower().endswith(".pdf") or fname.lower().endswith(".pdf")):
            continue
        url = encode_law_url(raw_path)
        if url in seen:
            continue
        seen.add(url)
        pdf_entries.append(
            {
                "url": url,
                "file_type": str(cf.get("type") or "content"),
                "file_name": fname or raw_path.rsplit("/", 1)[-1],
            }
        )
    return pdf_entries


def index_lawgoth_pages(
    session: requests.Session,
    catalog: LawCatalog,
    *,
    page_size: int = 50,
    delay: float = 0.25,
    max_pages: int | None = None,
) -> tuple[int, int]:
    """Paginate through apig.law.go.th/dga-user-service-phase2/law and index all PDF links."""
    last_page, _, completed = catalog.get_checkpoint("main")
    if completed:
        print("[index] law.go.th pagination already completed, skipping.")
        return 0, 0

    page = last_page + 1 if last_page > 0 else 1
    crawled = 0
    total_new_laws = 0
    total_new_pdfs = 0

    while True:
        if max_pages is not None and crawled >= max_pages:
            break

        payload = {
            "type": None,
            "agency": None,
            "hirachy": None,
            "size": page_size,
            "page": page,
        }
        try:
            resp = session.post(LAWGOTH_LIST_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"  [warn] page {page} failed: {exc}")
            break

        rows = data.get("rows") or []
        last_pg_api = int(data.get("lastPage") or 1)
        if not rows:
            catalog.save_checkpoint("main", page - 1, last_pg_api, completed=True)
            break

        page_new_pdfs = 0
        for law_row in rows:
            pdfs = extract_pdf_files(law_row)
            nl, nf = catalog.upsert_law_and_files(law_row, pdfs)
            total_new_laws += nl
            total_new_pdfs += nf
            page_new_pdfs += nf

        crawled += 1
        is_done = page >= last_pg_api
        catalog.save_checkpoint("main", page, last_pg_api, completed=is_done)
        if page == 1 or page % 10 == 0 or is_done:
            print(
                f"  [index] page {page}/{last_pg_api} "
                f"(rows={len(rows)}, +{page_new_pdfs} PDFs | total +{total_new_pdfs} PDFs)"
            )

        if is_done:
            break
        page += 1
        if delay > 0:
            time.sleep(delay)

    return total_new_laws, total_new_pdfs


def download_lawgoth_pdfs(
    catalog: LawCatalog,
    files_dir: Path,
    *,
    workers: int = 4,
    delay: float = 0.1,
    timeout: int = 35,
    min_bytes: int = 1024,
    max_file_bytes: int = 50 * 1024 * 1024,
    max_total_bytes: int = 0,
    max_files: int | None = None,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Download pending law.go.th PDFs concurrently with SHA-256 deduplication."""
    import pypdfium2 as pdfium

    files_dir.mkdir(parents=True, exist_ok=True)
    thread_local = threading.local()
    budget_lock = threading.Lock()
    in_flight_hashes: set[str] = set()
    current_bytes = catalog.get_total_downloaded_bytes()
    stats = {"downloaded": 0, "duplicate": 0, "skipped": 0, "failed": 0, "bytes_added": 0}

    def get_session() -> requests.Session:
        sess = getattr(thread_local, "session", None)
        if sess is None:
            sess = make_law_session(pool_maxsize=max(8, workers * 2))
            thread_local.session = sess
        return sess

    def download_one(item: dict) -> tuple[str, int]:
        nonlocal current_bytes
        url = item["url"]

        with budget_lock:
            if max_total_bytes > 0 and current_bytes >= max_total_bytes:
                return "budget_reached", 0

        if delay > 0:
            time.sleep(delay)

        sess = get_session()
        try:
            with sess.get(url, stream=True, allow_redirects=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    if resp.status_code in (404, 410):
                        catalog.mark_skipped(url, f"http_{resp.status_code}", http_status=resp.status_code)
                        return "skipped", 0
                    catalog.mark_failed(url, f"http_{resp.status_code}", http_status=resp.status_code)
                    return "failed", 0

                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_file_bytes:
                        catalog.mark_skipped(url, f"too_large:>{max_file_bytes}", http_status=200)
                        return "skipped", 0
                    chunks.append(chunk)

                if total < min_bytes:
                    catalog.mark_skipped(url, f"too_small:{total}", http_status=200)
                    return "skipped", 0
                raw = b"".join(chunks)
        except requests.RequestException as exc:
            catalog.mark_failed(url, str(exc))
            return "failed", 0

        if not raw.lstrip().startswith(b"%PDF-"):
            catalog.mark_skipped(url, "not_pdf_magic", http_status=200)
            return "skipped", 0

        try:
            doc = pdfium.PdfDocument(raw)
            try:
                num_pages = len(doc)
            finally:
                doc.close()
            if num_pages < 1:
                catalog.mark_skipped(url, "empty_pdf", http_status=200)
                return "skipped", 0
        except Exception as exc:
            catalog.mark_skipped(url, f"corrupt_pdf:{type(exc).__name__}", http_status=200)
            return "skipped", 0

        sha256 = hashlib.sha256(raw).hexdigest()
        bucket = files_dir / sha256[:2]
        dest_path = bucket / f"{sha256[:16]}.pdf"

        with budget_lock:
            existing = catalog.has_sha256(sha256)
            if existing or sha256 in in_flight_hashes:
                catalog.mark_duplicate(
                    url,
                    sha256=sha256,
                    size_bytes=total,
                    existing_path=existing or str(dest_path.as_posix()),
                )
                return "duplicate", 0
            if max_total_bytes > 0 and current_bytes + total > max_total_bytes:
                return "budget_reached", 0
            in_flight_hashes.add(sha256)
            current_bytes += total

        bucket.mkdir(parents=True, exist_ok=True)
        part_path = bucket / f"{sha256[:16]}.{threading.get_ident()}.pdf.part"
        try:
            if not dest_path.exists():
                part_path.write_bytes(raw)
                part_path.replace(dest_path)
        finally:
            part_path.unlink(missing_ok=True)
            with budget_lock:
                in_flight_hashes.discard(sha256)

        catalog.mark_downloaded(
            url,
            sha256=sha256,
            size_bytes=total,
            total_pages=num_pages,
            local_path=str(dest_path.as_posix()),
        )
        return "downloaded", total

    done_count = 0
    budget_hit = False
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while not budget_hit:
            if max_files is not None and done_count >= max_files:
                break
            limit = 100 if max_files is None else min(100, max_files - done_count)
            batch = catalog.fetch_pending_files(limit=limit, include_failed=retry_failed)
            if not batch:
                break
            futures = [pool.submit(download_one, item) for item in batch]
            for fut in as_completed(futures):
                outcome, added = fut.result()
                if outcome == "budget_reached":
                    budget_hit = True
                    continue
                stats[outcome] = stats.get(outcome, 0) + 1
                if outcome == "downloaded":
                    done_count += 1
                    stats["bytes_added"] += added

            mb_total = current_bytes / (1024 * 1024)
            print(
                f"  [download] +{stats['downloaded']} saved | {stats['duplicate']} dup | "
                f"{stats['skipped']} skip | {stats['failed']} fail | total={mb_total:.1f} MB"
            )

    return stats


def _print_stats(catalog: LawCatalog) -> None:
    st = catalog.get_stats()
    print(f"\n=== LawGoTh Catalog Stats ({catalog.db_path}) ===")
    print(f"Laws indexed: {st['laws_indexed']:,}")
    for status, info in sorted(st["by_status"].items()):
        print(f"  {status:12s}: {info['count']:6,d} files ({info['bytes'] / (1024**2):,.1f} MB)")


def run_collect_lawgoth(
    *,
    phase: str = "all",
    out_dir: Path = LAWGOTH_DIR,
    page_size: int = 50,
    workers: int = 4,
    index_delay: float = 0.2,
    download_delay: float = 0.1,
    max_pages: int | None = None,
    max_files: int | None = None,
    max_total_mb: int = 0,
    retry_failed: bool = False,
) -> dict:
    """Run indexing, PDF downloading, and manifest export for law.go.th."""
    out_dir = Path(out_dir)
    db_path = out_dir / "catalog.sqlite"
    files_dir = out_dir / "files"
    manifest_path = out_dir / "manifest.jsonl"

    result: dict = {}
    with LawCatalog(db_path) as catalog:
        if phase in {"all", "index"}:
            sess = make_law_session(pool_maxsize=8)
            nl, nf = index_lawgoth_pages(
                sess, catalog, page_size=page_size, delay=index_delay, max_pages=max_pages
            )
            result["indexed_laws"] = nl
            result["indexed_pdfs"] = nf
            print(f"[index] Done: +{nl:,} laws, +{nf:,} new PDF URLs")

        if phase in {"all", "download"}:
            dl_stats = download_lawgoth_pdfs(
                catalog,
                files_dir,
                workers=workers,
                delay=download_delay,
                max_total_bytes=max(0, max_total_mb) * 1024 * 1024,
                max_files=max_files,
                retry_failed=retry_failed,
            )
            result["download"] = dl_stats

        if phase in {"all", "export"}:
            exported = catalog.export_manifest(manifest_path)
            result["exported"] = exported
            print(f"[export] Wrote {exported:,} records to {manifest_path}")

        if phase in {"all", "stats"}:
            _print_stats(catalog)

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bulk PDF collector for law.go.th.")
    parser.add_argument("--phase", choices=["all", "index", "download", "export", "stats"], default="all")
    parser.add_argument("--out-dir", type=Path, default=LAWGOTH_DIR)
    parser.add_argument("--page-size", type=int, default=50, help="API page size (default: 50)")
    parser.add_argument("--workers", type=int, default=4, help="Download workers (default: 4)")
    parser.add_argument("--index-delay", type=float, default=0.2)
    parser.add_argument("--download-delay", type=float, default=0.1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-total-mb", type=int, default=0, help="Max total MB (0 = unlimited)")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)

    run_collect_lawgoth(
        phase=args.phase,
        out_dir=args.out_dir,
        page_size=args.page_size,
        workers=args.workers,
        index_delay=args.index_delay,
        download_delay=args.download_delay,
        max_pages=args.max_pages,
        max_files=args.max_files,
        max_total_mb=args.max_total_mb,
        retry_failed=args.retry_failed,
    )
