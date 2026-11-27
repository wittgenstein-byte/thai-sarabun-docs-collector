"""Tests for unified CLI subcommands and backward-compatible root shims."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from train_ocr.cli import SUBCOMMANDS


@pytest.mark.parametrize("subcommand", sorted(SUBCOMMANDS))
def test_cli_subcommand_help(subcommand: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "train_ocr.cli", subcommand, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    ("module_name", "expected_attr"),
    [
        ("config", "load_credentials"),
        ("saraban_utils", "has_thai_citizen_id"),
        ("eda_opendata", "run_eda"),
        ("collect_pdfs", "collect"),
        ("collect_dla", "collect_dla"),
        ("collect_dla_bulk", "run_bulk_harvest"),
        ("collect_lawgoth", "run_collect_lawgoth"),
        ("collect_saraban_extra", "collect_extra"),
        ("scan_pages", "scan_all"),
        ("scan_saraban_pages", "scan_saraban_all"),
        ("prepare_ocr_dataset", "build_dataset"),
        ("prepare_saraban_dataset", "build_saraban_dataset"),
        ("push_to_hf", "push_dataset"),
    ],
)
def test_root_shims_reexport_symbols(module_name: str, expected_attr: str) -> None:
    mod = importlib.import_module(module_name)
    assert hasattr(mod, expected_attr)
