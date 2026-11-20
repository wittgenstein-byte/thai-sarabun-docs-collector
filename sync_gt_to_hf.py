"""Merge verified GT annotations back into data/saraban/hf_dataset and optionally push to HF."""

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, Value

from config import (
    GOLDEN_MANIFEST_PATH,
    GT_ANNOTATIONS_PATH,
    HF_REPO_ID,
    SARABAN_HF_DIR,
    load_hf_token,
)


def sync_gt(push: bool = False) -> None:
    if not SARABAN_HF_DIR.exists():
        raise FileNotFoundError(f"Saraban HF dataset not found: {SARABAN_HF_DIR}")
    if not GT_ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"No GT annotations found at {GT_ANNOTATIONS_PATH}")

    golden_ids: set[str] = set()
    if GOLDEN_MANIFEST_PATH.exists():
        for line in GOLDEN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                golden_ids.add(json.loads(line)["sample_id"])

    annotations: dict[str, dict] = {}
    for line in GT_ANNOTATIONS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            annotations[rec["sample_id"]] = rec

    ds = DatasetDict.load_from_disk(str(SARABAN_HF_DIR))["test"]
    updated_rows: list[dict] = []
    verified_count = 0

    for i in range(len(ds)):
        row = dict(ds[i])
        sid = f"{row['resource_id']}_p{row['page_num']}"
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
    features["is_golden_set"] = Value("bool")
    features["gt_kie_json"] = Value("string")

    new_ds = Dataset.from_list(updated_rows, features=features)
    new_dict = DatasetDict({"test": new_ds})

    tmp_dir = SARABAN_HF_DIR.parent / "_hf_dataset_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    new_dict.save_to_disk(str(tmp_dir))
    shutil.rmtree(SARABAN_HF_DIR)
    tmp_dir.rename(SARABAN_HF_DIR)

    print(f"Synced GT into {SARABAN_HF_DIR}: {verified_count} human_verified / {len(golden_ids)} golden pages")

    if push:
        token = load_hf_token()
        new_dict.push_to_hub(
            HF_REPO_ID,
            config_name="saraban_6types",
            token=token,
            commit_message=f"Sync GT annotations ({verified_count} human_verified pages)",
        )
        print(f"Pushed updated saraban_6types to https://huggingface.co/datasets/{HF_REPO_ID}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync gt_annotations.jsonl into Saraban HF dataset.")
    parser.add_argument("--push", action="store_true", help="Also push updated dataset to Hugging Face Hub.")
    args = parser.parse_args()
    sync_gt(push=args.push)


if __name__ == "__main__":
    main()
