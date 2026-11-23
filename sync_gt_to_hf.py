"""Merge verified GT annotations back into HF dataset (saraban_6types or google_docs) and optionally push to HF."""

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, Value
from huggingface_hub import HfApi

from config import (
    GOLDEN_DIR,
    GOLDEN_MANIFEST_PATH,
    GT_ANNOTATIONS_PATH,
    HF_REPO_ID,
    SARABAN_HF_DIR,
    load_hf_token,
)


def export_groundtruth_backup() -> Path:
    """Export gt_annotations.jsonl and per-page .md files for text backup / version history."""
    backup_dir = GOLDEN_DIR / "groundtruth_backup"
    pages_dir = backup_dir / "pages"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(GT_ANNOTATIONS_PATH, backup_dir / "gt_annotations.jsonl")

    manifest_map: dict[str, dict] = {}
    if GOLDEN_MANIFEST_PATH.exists():
        for line in GOLDEN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                manifest_map[rec["sample_id"]] = rec

    md_count = 0
    for line in GT_ANNOTATIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ann = json.loads(line)
        sid = ann["sample_id"]
        gt_text = (ann.get("ground_truth_text") or "").strip()
        if not gt_text:
            continue
        meta = manifest_map.get(sid, {})
        st = meta.get("saraban_type", "unknown")
        out_dir = pages_dir / st
        out_dir.mkdir(parents=True, exist_ok=True)
        kie = ann.get("kie") or {}
        header = (
            f"<!-- sample_id: {sid} | saraban_type: {st} | page_type: {meta.get('page_type', '')} "
            f"| gt_status: {ann.get('gt_status', '')} | doc_no: {kie.get('doc_no', '')} "
            f"| doc_date: {kie.get('doc_date', '')} -->\n\n"
        )
        (out_dir / f"{sid}.md").write_text(header + gt_text + "\n", encoding="utf-8")
        md_count += 1

    print(f"Exported text backup: gt_annotations.jsonl + {md_count} .md files -> {backup_dir}")
    return backup_dir
from train_ocr.common.settings import GOOGLE_CRAWL_DIR, GOOGLE_HF_DIR


def sync_gt(dataset_name: str = "saraban", push: bool = False) -> None:
    is_google = dataset_name == "google_docs"
    hf_dir = GOOGLE_HF_DIR if is_google else SARABAN_HF_DIR
    ann_path = (GOOGLE_CRAWL_DIR / "gt" / "gt_annotations.jsonl") if is_google else GT_ANNOTATIONS_PATH
    config_name = "google_docs" if is_google else "saraban_6types"

    if not hf_dir.exists():
        raise FileNotFoundError(f"HF dataset not found: {hf_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"No GT annotations found at {ann_path}")

    golden_ids: set[str] = set()
    if not is_google and GOLDEN_MANIFEST_PATH.exists():
        for line in GOLDEN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                golden_ids.add(json.loads(line)["sample_id"])

    annotations: dict[str, dict] = {}
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            annotations[rec["sample_id"]] = rec

    ds = DatasetDict.load_from_disk(str(hf_dir))["test"]
    updated_rows: list[dict] = []
    verified_count = 0

    for i in range(len(ds)):
        row = dict(ds[i])
        sid = row["doc_id"] if is_google else f"{row['resource_id']}_p{row['page_num']}"
        if not is_google:
            row["is_golden_set"] = sid in golden_ids
        ann = annotations.get(sid)
        if ann:
            row["ground_truth_text"] = ann.get("ground_truth_text", "")
            row["gt_status"] = ann.get("gt_status", "unverified")
            row["quality_tags"] = ann.get("quality_tags", row.get("quality_tags", []))
            row["gt_kie_json"] = json.dumps(ann.get("kie") or {}, ensure_ascii=False)
            if row["gt_status"] == "human_verified":
                verified_count += 1
        else:
            row.setdefault("gt_kie_json", "{}")
        updated_rows.append(row)

    features = ds.features.copy()
    if not is_google:
        features["is_golden_set"] = Value("bool")
    features["gt_kie_json"] = Value("string")

    new_ds = Dataset.from_list(updated_rows, features=features)
    new_dict = DatasetDict({"test": new_ds})

    tmp_dir = hf_dir.parent / "_hf_dataset_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    new_dict.save_to_disk(str(tmp_dir))
    shutil.rmtree(hf_dir)
    tmp_dir.rename(hf_dir)

    total = len(ds) if is_google else len(golden_ids)
    print(f"Synced GT into {hf_dir}: {verified_count} human_verified / {total} pages")
    backup_dir = None if is_google else export_groundtruth_backup()

    if push:
        token = load_hf_token()
        new_dict.push_to_hub(
            HF_REPO_ID,
            config_name=config_name,
            token=token,
            commit_message=f"Sync {config_name} GT annotations ({verified_count} human_verified pages)",
        )
        if backup_dir is not None:
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(backup_dir),
                path_in_repo="groundtruth",
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload groundtruth/ backup ({verified_count} human_verified pages)",
            )
        print(f"Pushed updated {config_name} to https://huggingface.co/datasets/{HF_REPO_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync gt_annotations.jsonl into HF dataset.")
    parser.add_argument("--dataset", choices=["saraban", "google_docs"], default="saraban")
    parser.add_argument("--push", action="store_true", help="Also push updated dataset to Hugging Face Hub.")
    args = parser.parse_args()
    sync_gt(dataset_name=args.dataset, push=args.push)


if __name__ == "__main__":
    main()
