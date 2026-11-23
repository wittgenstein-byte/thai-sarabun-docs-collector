"""Full Thai OCR Benchmark Evaluation Suite: Strict/Norm CER, Thai WER, Table Cell/TEDS, KIE F1."""

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from pythainlp.tokenize import word_tokenize
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from config import (
    EVAL_REPORT_CSV_PATH,
    EVAL_REPORT_MD_PATH,
    GOLDEN_DRAFTS_PATH,
    GOLDEN_MANIFEST_PATH,
    GT_ANNOTATIONS_PATH,
)
from saraban_utils import normalize_thai_digits


def normalize_thai_ocr(text: str, strip_markdown: bool = False) -> str:
    """Normalize Thai text: NFC, Sara Am decomposition fix, Thai->Arabic digits, whitespace."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)
    # Fix decomposed Sara Am (nikahita + optional tone mark + sara aa -> optional tone mark + sara am)
    s = re.sub("\u0e4d([\u0e48-\u0e4b]?)\u0e32", "\\1\u0e33", s)
    s = normalize_thai_digits(s)
    if strip_markdown:
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"^\s*\|?[\s:-]+\|[\s|:-]*$", " ", s, flags=re.M)
        s = re.sub(r"[#*_`|~>]+", " ", s)
    return " ".join(s.split())


def compute_cer(ref: str, hyp: str) -> float:
    """Character Error Rate using Levenshtein distance."""
    if not ref and not hyp:
        return 0.0
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def compute_thai_wer(ref: str, hyp: str) -> float:
    """Word Error Rate using PyThaiNLP newmm word tokenizer."""
    ref_toks = [t for t in word_tokenize(ref, engine="newmm", keep_whitespace=False) if t.strip()]
    hyp_toks = [t for t in word_tokenize(hyp, engine="newmm", keep_whitespace=False) if t.strip()]
    if not ref_toks and not hyp_toks:
        return 0.0
    return Levenshtein.distance(ref_toks, hyp_toks) / max(len(ref_toks), 1)


def extract_tables(md_text: str) -> list[list[list[str]]]:
    """Extract HTML <table> and Markdown |...| tables into 2D cell grids."""
    tables: list[list[list[str]]] = []
    if not md_text:
        return tables

    # 1. HTML tables
    if "<table" in md_text.lower():
        soup = BeautifulSoup(md_text, "html.parser")
        for tbl in soup.find_all("table"):
            grid: list[list[str]] = []
            for tr in tbl.find_all("tr"):
                cells = [
                    normalize_thai_ocr(c.get_text(" ", strip=True))
                    for c in tr.find_all(["th", "td"])
                ]
                if cells:
                    grid.append(cells)
            if grid:
                tables.append(grid)

    # 2. Markdown pipe tables
    lines = md_text.splitlines()
    cur_block: list[str] = []
    for line in lines + [""]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) >= 3:
            cur_block.append(stripped)
        else:
            if len(cur_block) >= 2:
                grid = []
                for row_line in cur_block:
                    if re.match(r"^\|[\s:-]+(\|[\s:-]+)*\|$", row_line):
                        continue
                    cells = [normalize_thai_ocr(c.strip()) for c in row_line.strip("|").split("|")]
                    grid.append(cells)
                if grid:
                    tables.append(grid)
            cur_block = []

    return tables


def score_tables(ref_md: str, hyp_md: str) -> tuple[float | None, float | None]:
    """Compute (table_cell_f1, table_teds_approx) on pages that contain tables in GT."""
    ref_tables = extract_tables(ref_md)
    if not ref_tables:
        return None, None
    hyp_tables = extract_tables(hyp_md)
    if not hyp_tables:
        return 0.0, 0.0

    ref_cells = [c for tbl in ref_tables for row in tbl for c in row if c]
    hyp_cells = [c for tbl in hyp_tables for row in tbl for c in row if c]
    if not ref_cells:
        return None, None
    if not hyp_cells:
        return 0.0, 0.0

    # Cell-level F1 via greedy best match
    matched = 0.0
    used = set()
    for rc in ref_cells:
        best_score = 0.0
        best_j = -1
        for j, hc in enumerate(hyp_cells):
            if j in used:
                continue
            sc = fuzz.ratio(rc, hc) / 100.0
            if sc > best_score:
                best_score = sc
                best_j = j
        if best_j >= 0 and best_score >= 0.5:
            used.add(best_j)
            matched += best_score

    prec = matched / len(hyp_cells)
    rec = matched / len(ref_cells)
    cell_f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Approximate TEDS: average row/col shape similarity * cell_f1
    ref_rows = sum(len(t) for t in ref_tables)
    hyp_rows = sum(len(t) for t in hyp_tables)
    row_sim = 1.0 - abs(ref_rows - hyp_rows) / max(ref_rows, hyp_rows, 1)
    teds_approx = 0.4 * row_sim + 0.6 * cell_f1
    return round(cell_f1, 4), round(teds_approx, 4)


def evaluate_kie(gt_kie: dict, pred_kie: dict) -> tuple[float | None, float | None]:
    """Compute (kie_exact_acc, kie_norm_f1) over non-empty GT header fields."""
    fields = ("doc_no", "doc_date", "subject")
    active = [f for f in fields if (gt_kie.get(f) or "").strip()]
    if not active:
        return None, None

    exact_hits = 0
    f1_sum = 0.0
    for f in active:
        g = (gt_kie.get(f) or "").strip()
        p = (pred_kie.get(f) or "").strip()
        if g == p:
            exact_hits += 1
        gn = normalize_thai_ocr(g)
        pn = normalize_thai_ocr(p)
        f1_sum += fuzz.ratio(gn, pn) / 100.0

    return exact_hits / len(active), f1_sum / len(active)


def evaluate_model(
    model_name: str,
    gt_records: list[dict],
    preds_map: dict[str, dict],
) -> tuple[dict, list[dict]]:
    """Evaluate a single model's predictions against verified GT."""
    per_sample: list[dict] = []

    for gt in gt_records:
        sid = gt["sample_id"]
        ref_md = (gt.get("ground_truth_text") or "").strip()
        if not ref_md:
            continue
        pred = preds_map.get(sid, {})
        hyp_md = pred.get("pred_markdown", "")

        s_cer = compute_cer(ref_md, hyp_md)
        n_ref = normalize_thai_ocr(ref_md, strip_markdown=True)
        n_hyp = normalize_thai_ocr(hyp_md, strip_markdown=True)
        n_cer = compute_cer(n_ref, n_hyp)
        t_wer = compute_thai_wer(n_ref, n_hyp)
        cell_f1, teds = score_tables(ref_md, hyp_md)
        kie_exact, kie_f1 = evaluate_kie(gt.get("kie") or {}, pred.get("pred_kie") or {})

        per_sample.append(
            {
                "model": model_name,
                "sample_id": sid,
                "saraban_type": gt["saraban_type"],
                "page_type": gt["page_type"],
                "page_role": gt["page_role"],
                "quality_tags": gt.get("quality_tags", []),
                "strict_cer": s_cer,
                "norm_cer": n_cer,
                "thai_wer": t_wer,
                "table_cell_f1": cell_f1,
                "table_teds_approx": teds,
                "kie_exact_acc": kie_exact,
                "kie_norm_f1": kie_f1,
            }
        )

    def avg(vals: list[float | None]) -> float | None:
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    summary = {
        "model": model_name,
        "n_samples": len(per_sample),
        "strict_cer": avg([x["strict_cer"] for x in per_sample]),
        "norm_cer": avg([x["norm_cer"] for x in per_sample]),
        "thai_wer": avg([x["thai_wer"] for x in per_sample]),
        "table_cell_f1": avg([x["table_cell_f1"] for x in per_sample]),
        "table_teds_approx": avg([x["table_teds_approx"] for x in per_sample]),
        "kie_exact_acc": avg([x["kie_exact_acc"] for x in per_sample]),
        "kie_norm_f1": avg([x["kie_norm_f1"] for x in per_sample]),
    }
    return summary, per_sample


def format_pct_or_na(val: float | None) -> str:
    return f"{val * 100:.2f}%" if val is not None else "N/A"


def write_reports(summaries: list[dict], all_samples: list[dict]) -> None:
    """Write Markdown leaderboard + slice breakdown and CSV output."""
    lines = [
        "# Thai Saraban OCR Benchmark Evaluation Report\n",
        "## Overall Leaderboard\n",
        "| Model | N | Strict CER (↓) | Norm CER (↓) | Thai WER (↓) | Table Cell F1 (↑) | Table TEDS (↑) | KIE Exact (↑) | KIE Norm F1 (↑) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(summaries, key=lambda x: x["norm_cer"] or 999):
        lines.append(
            f"| `{s['model']}` | {s['n_samples']} | "
            f"{format_pct_or_na(s['strict_cer'])} | {format_pct_or_na(s['norm_cer'])} | "
            f"{format_pct_or_na(s['thai_wer'])} | {format_pct_or_na(s['table_cell_f1'])} | "
            f"{format_pct_or_na(s['table_teds_approx'])} | {format_pct_or_na(s['kie_exact_acc'])} | "
            f"{format_pct_or_na(s['kie_norm_f1'])} |"
        )

    for slice_col in ("saraban_type", "page_type", "page_role"):
        lines.append(f"\n## Breakdown by `{slice_col}` (Norm CER ↓)\n")
        models = [s["model"] for s in summaries]
        header = f"| `{slice_col}` | " + " | ".join(f"`{m}`" for m in models) + " |"
        sep = "|---|" + "|".join("---:" for _ in models) + "|"
        lines.extend([header, sep])

        slice_vals = sorted({r[slice_col] for r in all_samples})
        for sv in slice_vals:
            row_cells = [f"`{sv}`"]
            for m in models:
                sub = [r["norm_cer"] for r in all_samples if r["model"] == m and r[slice_col] == sv]
                row_cells.append(format_pct_or_na(sum(sub) / len(sub) if sub else None))
            lines.append("| " + " | ".join(row_cells) + " |")

    EVAL_REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with EVAL_REPORT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\n".join(lines))
    print(f"\nSaved reports: {EVAL_REPORT_MD_PATH} and {EVAL_REPORT_CSV_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR models against human_verified GT.")
    parser.add_argument(
        "--preds",
        nargs="*",
        default=[],
        help="Model predictions as name=path/to/preds.jsonl",
    )
    parser.add_argument(
        "--eval-drafts",
        action="store_true",
        help="Evaluate Gemini and iApp OpenThai2.1 drafts directly against verified GT.",
    )
    args = parser.parse_args()

    if not GOLDEN_MANIFEST_PATH.exists() or not GT_ANNOTATIONS_PATH.exists():
        raise FileNotFoundError("Missing golden_manifest.jsonl or gt_annotations.jsonl")

    manifest_map = {
        rec["sample_id"]: rec
        for line in GOLDEN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for rec in [json.loads(line)]
    }

    gt_records: list[dict] = []
    for line in GT_ANNOTATIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ann = json.loads(line)
        if ann.get("gt_status") != "human_verified":
            continue
        m = manifest_map.get(ann["sample_id"], {})
        gt_records.append({**m, **ann})

    if not gt_records:
        raise RuntimeError("No human_verified GT rows found in gt_annotations.jsonl")

    model_preds: dict[str, dict[str, dict]] = {}
    if args.eval_drafts and GOLDEN_DRAFTS_PATH.exists():
        map_a, map_b = {}, {}
        name_a, name_b = "model_a", "model_b"
        for line in GOLDEN_DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            sid = d["sample_id"]
            name_a = d.get("model_a_name", name_a)
            name_b = d.get("model_b_name", name_b)
            map_a[sid] = {
                "pred_markdown": d.get("model_a_markdown") or d.get("gemini_markdown") or "",
                "pred_kie": d.get("model_a_kie") or d.get("gemini_kie") or {},
            }
            map_b[sid] = {
                "pred_markdown": d.get("model_b_markdown") or d.get("iapp_markdown") or "",
                "pred_kie": d.get("model_b_kie") or {},
            }
        model_preds[name_a] = map_a
        model_preds[name_b] = map_b

    for spec in args.preds:
        name, path_str = spec.split("=", 1)
        pmap = {}
        for line in Path(path_str).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                pmap[r["sample_id"]] = r
        model_preds[name] = pmap

    if not model_preds:
        raise ValueError("Specify --eval-drafts or --preds model=path.jsonl")

    summaries, all_samples = [], []
    for mname, pmap in model_preds.items():
        s, rows = evaluate_model(mname, gt_records, pmap)
        summaries.append(s)
        all_samples.extend(rows)

    write_reports(summaries, all_samples)


if __name__ == "__main__":
    main()
