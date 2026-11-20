"""SQLite WAL catalog and checkpoint store for the DLA bulk document harvester."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DlaCatalog:
    """Thread-safe SQLite catalog tracking crawl checkpoints, documents, and file downloads."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS crawl_state (
                source TEXT NOT NULL,
                subkey TEXT NOT NULL,
                last_position INTEGER NOT NULL DEFAULT 0,
                total_records INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source, subkey)
            );

            CREATE TABLE IF NOT EXISTS documents (
                doc_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                subkey TEXT NOT NULL,
                dept TEXT NOT NULL DEFAULT '',
                doc_no TEXT NOT NULL DEFAULT '',
                doc_date TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                page_url TEXT NOT NULL DEFAULT '',
                extra_json TEXT NOT NULL DEFAULT '{}',
                discovered_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                url TEXT PRIMARY KEY,
                doc_key TEXT NOT NULL,
                source TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                original_filename TEXT NOT NULL DEFAULT '',
                ext TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                sha256 TEXT NOT NULL DEFAULT '',
                size_bytes INTEGER NOT NULL DEFAULT 0,
                total_pages INTEGER NOT NULL DEFAULT 0,
                local_path TEXT NOT NULL DEFAULT '',
                http_status INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                retries INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (doc_key) REFERENCES documents(doc_key)
            );

            CREATE INDEX IF NOT EXISTS idx_files_status_source ON files(status, source);
            CREATE INDEX IF NOT EXISTS idx_files_sha256 ON files(sha256);
            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> DlaCatalog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_checkpoint(self, source: str, subkey: str = "") -> tuple[int, int, bool]:
        """Return (last_position, total_records, completed) for a source/subkey."""
        with self._lock:
            row = self._conn.execute(
                "SELECT last_position, total_records, completed FROM crawl_state WHERE source=? AND subkey=?",
                (source, subkey),
            ).fetchone()
            if row is None:
                return 0, 0, False
            return int(row["last_position"]), int(row["total_records"]), bool(row["completed"])

    def save_checkpoint(
        self,
        source: str,
        subkey: str,
        last_position: int,
        total_records: int = 0,
        completed: bool = False,
    ) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO crawl_state (source, subkey, last_position, total_records, completed, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, subkey) DO UPDATE SET
                    last_position=excluded.last_position,
                    total_records=MAX(crawl_state.total_records, excluded.total_records),
                    completed=excluded.completed,
                    updated_at=excluded.updated_at
                """,
                (source, subkey, last_position, total_records, int(completed), now),
            )
            self._conn.commit()

    def upsert_document_and_files(
        self,
        doc: dict,
        file_entries: Iterable[dict],
    ) -> tuple[int, int]:
        """Insert or ignore a document and its associated file URLs.

        Returns (new_docs_count, new_files_count).
        """
        now = _utc_now()
        extra_json = json.dumps(doc.get("extra") or {}, ensure_ascii=False)
        with self._lock:
            cur_doc = self._conn.execute(
                """
                INSERT OR IGNORE INTO documents
                    (doc_key, source, subkey, dept, doc_no, doc_date, title, page_url, extra_json, discovered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc["doc_key"],
                    doc["source"],
                    doc.get("subkey", ""),
                    doc.get("dept", ""),
                    doc.get("doc_no", ""),
                    doc.get("doc_date", ""),
                    doc.get("title", ""),
                    doc.get("page_url", ""),
                    extra_json,
                    now,
                ),
            )
            new_files = 0
            for f in file_entries:
                cur_f = self._conn.execute(
                    """
                    INSERT OR IGNORE INTO files
                        (url, doc_key, source, label, original_filename, ext, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        f["url"],
                        doc["doc_key"],
                        doc["source"],
                        f.get("label", ""),
                        f.get("original_filename", ""),
                        f.get("ext", ""),
                        now,
                    ),
                )
                new_files += cur_f.rowcount
            self._conn.commit()
            return cur_doc.rowcount, new_files

    def fetch_pending_files(
        self,
        limit: int = 200,
        sources: list[str] | None = None,
        allowed_exts: tuple[str, ...] | None = None,
        include_failed: bool = False,
        max_retries: int = 3,
    ) -> list[dict]:
        """Fetch a batch of pending (and optionally retryable failed) file rows."""
        statuses = ["'pending'"]
        if include_failed:
            statuses.append("'failed'")
        where_parts = [f"status IN ({','.join(statuses)})", "retries < ?"]
        params: list[object] = [max_retries]

        if sources:
            placeholders = ",".join("?" for _ in sources)
            where_parts.append(f"source IN ({placeholders})")
            params.extend(sources)

        if allowed_exts:
            norm_exts = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in allowed_exts]
            placeholders = ",".join("?" for _ in norm_exts)
            where_parts.append(f"(LOWER(ext) IN ({placeholders}) OR ext = '')")
            params.extend(norm_exts)

        sql = f"""
            SELECT url, doc_key, source, label, original_filename, ext, retries
            FROM files
            WHERE {' AND '.join(where_parts)}
            ORDER BY retries ASC, updated_at ASC
            LIMIT ?
        """
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def has_sha256(self, sha256: str) -> str | None:
        """Return existing local_path if sha256 was already downloaded, else None."""
        if not sha256:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT local_path FROM files WHERE sha256=? AND status='downloaded' AND local_path != '' LIMIT 1",
                (sha256,),
            ).fetchone()
            return str(row["local_path"]) if row else None

    def get_total_downloaded_bytes(self) -> int:
        """Return total size in bytes of all unique downloaded files."""
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
        ext: str,
        http_status: int = 200,
    ) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE files
                SET status='downloaded', sha256=?, size_bytes=?, total_pages=?,
                    local_path=?, ext=?, http_status=?, error='', updated_at=?
                WHERE url=?
                """,
                (sha256, size_bytes, total_pages, local_path, ext, http_status, now, url),
            )
            self._conn.commit()

    def mark_duplicate(
        self,
        url: str,
        *,
        sha256: str,
        size_bytes: int,
        existing_path: str,
        ext: str,
        http_status: int = 200,
    ) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE files
                SET status='duplicate', sha256=?, size_bytes=?, local_path=?,
                    ext=?, http_status=?, error='', updated_at=?
                WHERE url=?
                """,
                (sha256, size_bytes, existing_path, ext, http_status, now, url),
            )
            self._conn.commit()

    def mark_skipped(self, url: str, reason: str, http_status: int = 0) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE files
                SET status='skipped', error=?, http_status=?, updated_at=?
                WHERE url=?
                """,
                (reason[:300], http_status, now, url),
            )
            self._conn.commit()

    def mark_failed(self, url: str, error: str, http_status: int = 0) -> None:
        now = _utc_now()
        with self._lock:
            self._conn.execute(
                """
                UPDATE files
                SET status='failed', error=?, http_status=?, retries=retries+1, updated_at=?
                WHERE url=?
                """,
                (error[:300], http_status, now, url),
            )
            self._conn.commit()

    def export_manifest_jsonl(self, manifest_path: Path) -> int:
        """Export all downloaded files joined with document metadata to JSONL."""
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    f.url AS source_url,
                    f.source,
                    f.label AS link_label,
                    f.original_filename,
                    f.ext,
                    f.sha256,
                    f.size_bytes,
                    f.total_pages,
                    f.local_path,
                    d.doc_key,
                    d.subkey,
                    d.dept,
                    d.doc_no,
                    d.doc_date,
                    d.title,
                    d.page_url
                FROM files f
                JOIN documents d ON f.doc_key = d.doc_key
                WHERE f.status = 'downloaded'
                ORDER BY f.source, d.doc_date DESC, f.url
                """
            ).fetchall()

        count = 0
        with manifest_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                rec = dict(row)
                rec["doc_id"] = rec["sha256"][:16]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
        return count

    def get_stats(self) -> dict:
        """Return summary statistics across documents, files, sources, and extensions."""
        with self._lock:
            doc_total = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            by_status = {
                r["status"]: {"count": r["cnt"], "bytes": r["bytes"]}
                for r in self._conn.execute(
                    "SELECT status, COUNT(*) AS cnt, COALESCE(SUM(size_bytes), 0) AS bytes FROM files GROUP BY status"
                ).fetchall()
            }
            by_source = [
                dict(r)
                for r in self._conn.execute(
                    """
                    SELECT source,
                           COUNT(*) AS total_files,
                           SUM(CASE WHEN status='downloaded' THEN 1 ELSE 0 END) AS downloaded,
                           SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                           SUM(CASE WHEN status='duplicate' THEN 1 ELSE 0 END) AS duplicate,
                           SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) AS skipped,
                           SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                           COALESCE(SUM(CASE WHEN status='downloaded' THEN size_bytes ELSE 0 END), 0) AS bytes
                    FROM files
                    GROUP BY source
                    ORDER BY total_files DESC
                    """
                ).fetchall()
            ]
            by_ext = [
                dict(r)
                for r in self._conn.execute(
                    """
                    SELECT LOWER(ext) AS ext,
                           COUNT(*) AS count,
                           COALESCE(SUM(size_bytes), 0) AS bytes
                    FROM files
                    WHERE status = 'downloaded'
                    GROUP BY LOWER(ext)
                    ORDER BY count DESC
                    """
                ).fetchall()
            ]
        return {
            "documents": int(doc_total),
            "by_status": by_status,
            "by_source": by_source,
            "by_ext": by_ext,
        }
