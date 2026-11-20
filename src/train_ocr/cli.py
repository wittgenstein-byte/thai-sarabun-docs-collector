"""Unified CLI entrypoint (`train-ocr`) with lazy per-stage dispatching."""

from __future__ import annotations

import argparse
import importlib
import sys

SUBCOMMANDS: dict[str, tuple[str, str, str]] = {
    "eda": (
        "train_ocr.collectors.datagoth",
        "main_eda",
        "Run Exploratory Data Analysis against data.go.th CKAN API",
    ),
    "collect-general": (
        "train_ocr.collectors.datagoth",
        "main_general",
        "Collect diverse Thai government PDFs from data.go.th",
    ),
    "collect-dla": (
        "train_ocr.collectors.dla",
        "main",
        "Collect 6-type Thai Saraban PDFs from dla.go.th",
    ),
    "collect-dla-bulk": (
        "train_ocr.collectors.dla_bulk",
        "main",
        "Bulk-harvest all public PDFs & images from dla.go.th and sll.dla.go.th",
    ),
    "collect-saraban-extra": (
        "train_ocr.collectors.datagoth",
        "main_saraban_extra",
        "Collect supplementary Saraban PDFs from data.go.th",
    ),
    "collect-google": (
        "train_ocr.collectors.google_images",
        "main",
        "Collect Thai government document images via AutoCrawler adapter",
    ),
    "scan-general": (
        "train_ocr.screener.pdf_scanner",
        "main_general",
        "Scan general data.go.th PDFs page-by-page",
    ),
    "scan-saraban": (
        "train_ocr.screener.pdf_scanner",
        "main_saraban",
        "Scan Saraban PDFs page-by-page",
    ),
    "filter-images": (
        "train_ocr.screener.filter_images",
        "main",
        "Filter crawled images into kept/ and rejected/ for OCR quality",
    ),
    "serve-screener": (
        "train_ocr.cli",
        "_serve_screener",
        "Run the FastAPI ocr-screener microservice",
    ),
    "build-general": (
        "train_ocr.dataset.builder",
        "main_general",
        "Build stratified general OCR benchmark HF dataset",
    ),
    "build-saraban": (
        "train_ocr.dataset.builder",
        "main_saraban",
        "Build 300 DPI stratified 6-type Saraban OCR benchmark HF dataset",
    ),
    "push-hf": (
        "train_ocr.publisher.hf_hub",
        "main",
        "Push prepared benchmark dataset and README to Hugging Face Hub",
    ),
}


def _serve_screener(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="train-ocr serve-screener", description="Serve ocr-screener API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    uvicorn.run("train_ocr.screener.api:app", host=args.host, port=args.port)


def _format_help() -> str:
    lines = [
        "usage: train-ocr <command> [args...]",
        "",
        "Available commands:",
    ]
    for name, (_, _, desc) in SUBCOMMANDS.items():
        lines.append(f"  {name:22s} {desc}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_format_help())
        return

    cmd, rest = args[0], args[1:]
    if cmd not in SUBCOMMANDS:
        print(f"Unknown command: {cmd}\n\n{_format_help()}", file=sys.stderr)
        raise SystemExit(2)

    module_name, func_name, _ = SUBCOMMANDS[cmd]
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    func(rest)


if __name__ == "__main__":
    main()
