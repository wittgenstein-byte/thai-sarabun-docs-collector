"""High-coverage, resumable bulk document harvester for dla.go.th and sll.dla.go.th."""

from __future__ import annotations

import argparse
from pathlib import Path

from train_ocr.collectors.dla_bulk.catalog import DlaCatalog
from train_ocr.collectors.dla_bulk.downloader import run_downloader
from train_ocr.collectors.dla_bulk.portals import ALL_SOURCES, CRAWLERS, make_session
from train_ocr.common.settings import (
    DLA_BULK_DEFAULT_EXTS,
    DLA_BULK_DIR,
    DLA_BULK_MAX_FILE_BYTES,
    DLA_BULK_MAX_TOTAL_BYTES,
)


def _print_stats(catalog: DlaCatalog) -> None:
    stats = catalog.get_stats()
    print(f"\n=== DLA Bulk Catalog Stats ({catalog.db_path}) ===")
    print(f"Documents indexed : {stats['documents']:,}")
    for status, info in sorted(stats["by_status"].items()):
        mb = info["bytes"] / (1024 * 1024)
        print(f"  {status:12s}: {info['count']:8,d} files ({mb:,.1f} MB)")
    if stats["by_source"]:
        print("\nBy source:")
        for row in stats["by_source"]:
            mb = row["bytes"] / (1024 * 1024)
            print(
                f"  {row['source']:14s} | total={row['total_files']:6,d} "
                f"dl={row['downloaded']:6,d} pend={row['pending']:6,d} "
                f"dup={row['duplicate']:4,d} skip={row['skipped']:4,d} fail={row['failed']:4,d} "
                f"({mb:,.1f} MB)"
            )
    if stats["by_ext"]:
        print("\nDownloaded by extension:")
        for row in stats["by_ext"]:
            mb = row["bytes"] / (1024 * 1024)
            print(f"  {row['ext'] or '(none)':8s}: {row['count']:6,d} files ({mb:,.1f} MB)")


def run_bulk_harvest(
    *,
    sources: list[str] | None = None,
    phase: str = "all",
    out_dir: Path = DLA_BULK_DIR,
    allowed_exts: tuple[str, ...] | None = DLA_BULK_DEFAULT_EXTS,
    workers: int = 4,
    index_delay: float = 0.5,
    download_delay: float = 0.15,
    max_pages: int | None = None,
    max_files: int | None = None,
    max_file_bytes: int = DLA_BULK_MAX_FILE_BYTES,
    max_total_bytes: int = DLA_BULK_MAX_TOTAL_BYTES,
    skip_pii: bool = False,
    retry_failed: bool = False,
) -> dict:
    """Run the DLA bulk indexing, downloading, and manifest export pipeline."""
    out_dir = Path(out_dir)
    db_path = out_dir / "catalog.sqlite"
    files_dir = out_dir / "files"
    manifest_path = out_dir / "manifest.jsonl"

    target_sources = list(ALL_SOURCES) if not sources else sources
    for src in target_sources:
        if src not in CRAWLERS:
            raise ValueError(f"Unknown source '{src}'. Valid: {', '.join(ALL_SOURCES)}")

    result: dict = {}
    with DlaCatalog(db_path) as catalog:
        if phase in {"all", "index"}:
            session = make_session(pool_maxsize=8)
            indexed_docs = 0
            indexed_files = 0
            print(f"[index] Crawling {len(target_sources)} portal(s): {', '.join(target_sources)}")
            for src in target_sources:
                nd, nf = CRAWLERS[src](
                    session,
                    catalog,
                    delay=index_delay,
                    max_pages=max_pages,
                    skip_pii=skip_pii,
                )
                indexed_docs += nd
                indexed_files += nf
            result["indexed_docs"] = indexed_docs
            result["indexed_files"] = indexed_files
            print(f"[index] Done: +{indexed_docs:,} docs, +{indexed_files:,} new file URLs")

        if phase in {"all", "download"}:
            print(
                f"[download] workers={workers}, allowed_exts={allowed_exts or 'ALL'}, "
                f"budget={max_total_bytes / (1024**2):.0f} MB"
            )
            dl_stats = run_downloader(
                catalog,
                files_dir,
                sources=target_sources if sources else None,
                allowed_exts=allowed_exts,
                workers=workers,
                delay=download_delay,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                max_files=max_files,
                retry_failed=retry_failed,
            )
            result["download"] = dl_stats

        if phase in {"all", "export"}:
            exported = catalog.export_manifest_jsonl(manifest_path)
            result["exported"] = exported
            print(f"[export] Wrote {exported:,} records to {manifest_path}")

        if phase in {"all", "stats"}:
            _print_stats(catalog)

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bulk document harvester for www.dla.go.th and sll.dla.go.th."
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help=f"Comma-separated sources or 'all'. Choices: {', '.join(ALL_SOURCES)}",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "index", "download", "export", "stats"],
        default="all",
        help="Pipeline phase to execute (default: all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DLA_BULK_DIR,
        help=f"Output directory for catalog.sqlite and files/ (default: {DLA_BULK_DIR})",
    )
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="Download PDFs only (default: PDFs + images)",
    )
    parser.add_argument(
        "--all-types",
        action="store_true",
        help="Download all file extensions instead of only PDFs + images",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download workers (default: 4)")
    parser.add_argument("--index-delay", type=float, default=0.5, help="Delay between index pages (sec)")
    parser.add_argument("--download-delay", type=float, default=0.15, help="Delay per file download (sec)")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages per source to crawl")
    parser.add_argument("--max-files", type=int, default=None, help="Max files to download in this run")
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=DLA_BULK_MAX_FILE_BYTES // (1024 * 1024),
        help="Max single file size in MB (default: 50)",
    )
    parser.add_argument(
        "--max-total-mb",
        type=int,
        default=DLA_BULK_MAX_TOTAL_BYTES // (1024 * 1024),
        help="Max total downloaded storage budget in MB (default: 1024 = 1 GB; 0 = unlimited)",
    )
    parser.add_argument(
        "--skip-pii",
        action="store_true",
        help="Skip documents matching PII_TITLE_BLOCKLIST (default: keep all)",
    )
    parser.add_argument("--retry-failed", action="store_true", help="Include previously failed URLs")

    args = parser.parse_args(argv)
    sources = None if args.sources.strip().lower() == "all" else [
        s.strip() for s in args.sources.split(",") if s.strip()
    ]

    if args.all_types:
        allowed_exts = None
    elif args.pdf_only:
        allowed_exts = (".pdf",)
    else:
        allowed_exts = DLA_BULK_DEFAULT_EXTS

    run_bulk_harvest(
        sources=sources,
        phase=args.phase,
        out_dir=args.out_dir,
        allowed_exts=allowed_exts,
        workers=args.workers,
        index_delay=args.index_delay,
        download_delay=args.download_delay,
        max_pages=args.max_pages,
        max_files=args.max_files,
        max_file_bytes=max(1, args.max_file_mb) * 1024 * 1024,
        max_total_bytes=max(0, args.max_total_mb) * 1024 * 1024,
        skip_pii=args.skip_pii,
        retry_failed=args.retry_failed,
    )


__all__ = ["ALL_SOURCES", "DlaCatalog", "main", "run_bulk_harvest"]
