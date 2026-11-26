"""Collectors and EDA for Thailand Open Data (data.go.th) REST API."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests

from train_ocr.common.manifest import load_saraban_manifest, read_jsonl
from train_ocr.common.pdf_io import download_and_validate_pdf, has_scanned_pages
from train_ocr.common.saraban_rules import is_pii_blocked_title
from train_ocr.common.settings import (
    ALLOWED_LICENSES,
    BASE_URL,
    GROUPS,
    MANIFEST_PATH,
    MAX_PER_ORG,
    RAW_PDF_DIR,
    REQUEST_TIMEOUT,
    SARABAN_MANIFEST_PATH,
    SARABAN_RAW_DIR,
    TARGET_PER_GROUP,
    load_datagoth_api_key,
)

DATA_GOTH_QUERIES = [
    ("order_regulation", "order", "คำสั่ง"),
    ("order_regulation", "regulation", "ระเบียบ"),
    ("order_regulation", "bylaw", "ข้อบังคับ"),
    ("public_relations", "announcement", "ประกาศ"),
    ("public_relations", "press_release", "ข่าวประชาสัมพันธ์"),
    ("official_record", "meeting_minutes", "รายงานการประชุม"),
    ("official_record", "certificate_template", "หนังสือรับรอง"),
]


def load_existing_manifest(
    manifest_path: Path,
) -> tuple[set[str], set[str], dict[str, int], dict[str, int]]:
    """Load already collected resource IDs, SHA256 hashes, and per-group/org counts."""
    seen_resources: set[str] = set()
    seen_hashes: set[str] = set()
    group_counts: dict[str, int] = {g: 0 for g in GROUPS}
    org_counts: dict[str, int] = {}

    for rec in read_jsonl(manifest_path):
        seen_resources.add(rec["resource_id"])
        seen_hashes.add(rec["sha256"])
        grp = rec["group"]
        group_counts[grp] = group_counts.get(grp, 0) + 1
        org = rec["organization_id"]
        org_counts[org] = org_counts.get(org, 0) + 1

    return seen_resources, seen_hashes, group_counts, org_counts


def fetch_group_candidates(
    session: requests.Session,
    api_key: str,
    group: str,
    rows: int = 120,
    start: int = 0,
    query: str | None = None,
) -> list[dict]:
    """Query CKAN package_search for PDF datasets in a specific group."""
    params: dict[str, str | int] = {
        "fq": f"res_format:PDF AND groups:{group}",
        "rows": rows,
        "start": start,
    }
    if query:
        params["q"] = query
    for attempt in range(2):
        try:
            resp = session.get(
                f"{BASE_URL}/package_search",
                headers={"api-key": api_key},
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success"):
                return []
            return data["result"]["results"]
        except requests.RequestException as exc:
            if attempt == 0:
                import time
                time.sleep(2.0)
                continue
            print(f"  [warn] API query failed ({group}, start={start}): {exc}")
            return []
    return []


def download_and_validate_general_pdf(
    session: requests.Session, url: str, dest_path: Path, scanned_only: bool = False
) -> tuple[bool, str, int, int]:
    """Backward-compatible wrapper returning (ok, sha256, size_bytes, total_pages)."""
    ok, sha256, size_bytes, total_pages, _ = download_and_validate_pdf(
        session,
        url,
        dest_path=dest_path,
        scanned_only=scanned_only,
        timeout=REQUEST_TIMEOUT,
        check_head=True,
    )
    return ok, sha256, size_bytes, total_pages


def collect_general(
    per_group: int = TARGET_PER_GROUP,
    max_per_org: int = MAX_PER_ORG,
    seed: int = 42,
    scanned_only: bool = False,
    target_new: int = 100,
) -> None:
    """Harvest diverse Thai government PDFs from data.go.th across all 15 groups."""
    api_key = load_datagoth_api_key()
    rng = random.Random(seed)

    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_resources, seen_hashes, group_counts, org_counts = load_existing_manifest(MANIFEST_PATH)

    session = requests.Session()
    session.headers.update({"User-Agent": "thai-govern-ocr-benchmark/1.0"})

    added_new = 0
    start_offsets = tuple(range(0, 1200, 120)) if scanned_only else (0, 120)

    with MANIFEST_PATH.open("a", encoding="utf-8") as mf:
        for group in GROUPS:
            if scanned_only and added_new >= target_new:
                break

            current = group_counts.get(group, 0)
            if not scanned_only and current >= per_group:
                print(f"[{group}] already has {current}/{per_group} PDFs, skipping.")
                continue

            print(f"[{group}] collecting (currently {current}, scanned_only={scanned_only})...")
            candidates: list[dict] = []
            seen_pkg_ids: set[str] = set()
            for start in start_offsets:
                batch = fetch_group_candidates(session, api_key, group, rows=120, start=start)
                for pkg in batch:
                    if pkg["id"] not in seen_pkg_ids:
                        seen_pkg_ids.add(pkg["id"])
                        candidates.append(pkg)
                if len(batch) < 120:
                    break

            rng.shuffle(candidates)
            group_new = 0
            group_cap = max(1, (target_new // len(GROUPS)) + 2) if scanned_only else per_group

            for pkg in candidates:
                if scanned_only and (added_new >= target_new or group_new >= group_cap):
                    break
                if not scanned_only and group_counts.get(group, 0) >= per_group:
                    break

                license_id = pkg.get("license_id") or ""
                if license_id not in ALLOWED_LICENSES:
                    continue
                if is_pii_blocked_title(pkg.get("title") or ""):
                    continue

                org_obj = pkg.get("organization") or {}
                org_id = org_obj.get("name") or "unknown"
                org_title = org_obj.get("title") or org_id
                if org_counts.get(org_id, 0) >= max_per_org:
                    continue

                pdf_resources = [
                    r
                    for r in pkg.get("resources", [])
                    if (r.get("format") or "").strip().upper() == "PDF" and r.get("url")
                ]
                if not pdf_resources:
                    continue

                res = rng.choice(pdf_resources)
                res_id = res["id"]
                if res_id in seen_resources:
                    continue

                dest_path = RAW_PDF_DIR / group / f"{res_id}.pdf"
                ok, sha256, size_bytes, total_pages, _ = download_and_validate_pdf(
                    session,
                    res["url"],
                    dest_path=dest_path,
                    scanned_only=scanned_only,
                    timeout=REQUEST_TIMEOUT,
                    check_head=True,
                )
                if not ok:
                    continue

                if sha256 in seen_hashes:
                    dest_path.unlink(missing_ok=True)
                    continue

                record = {
                    "doc_id": pkg["id"],
                    "resource_id": res_id,
                    "group": group,
                    "organization_id": org_id,
                    "organization": org_title,
                    "title": pkg.get("title") or "",
                    "resource_name": res.get("name") or "",
                    "license_id": license_id,
                    "source_url": res["url"],
                    "metadata_modified": (pkg.get("metadata_modified") or "")[:10],
                    "local_path": str(dest_path.as_posix()),
                    "sha256": sha256,
                    "size_bytes": size_bytes,
                    "total_pages": total_pages,
                }
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                mf.flush()

                seen_resources.add(res_id)
                seen_hashes.add(sha256)
                group_counts[group] = group_counts.get(group, 0) + 1
                org_counts[org_id] = org_counts.get(org_id, 0) + 1
                added_new += 1
                group_new += 1
                print(
                    f"  + [{group} +{group_new} | total_new={added_new}] "
                    f"{org_title[:30]} | {total_pages}p | {size_bytes // 1024}KB"
                )

    total_docs = sum(group_counts.values())
    print(
        f"\nCollection complete: added {added_new} new PDFs "
        f"(total {total_docs} PDFs across {len(org_counts)} orgs)."
    )


def collect_saraban_extra(per_query: int = 10) -> None:
    """Collect supplementary cross-agency Saraban PDFs from data.go.th."""
    api_key = load_datagoth_api_key()
    SARABAN_RAW_DIR.mkdir(parents=True, exist_ok=True)
    SARABAN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_urls, seen_hashes, type_counts, _ = load_saraban_manifest(SARABAN_MANIFEST_PATH)

    session = requests.Session()
    session.headers.update({"User-Agent": "thai-saraban-ocr-benchmark/1.0"})

    added = 0
    with SARABAN_MANIFEST_PATH.open("a", encoding="utf-8") as mf:
        for saraban_type, subtype, kw in DATA_GOTH_QUERIES:
            try:
                resp = session.get(
                    f"{BASE_URL}/package_search",
                    headers={"api-key": api_key},
                    params={"fq": "res_format:PDF", "q": kw, "rows": 80},
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                pkgs = resp.json().get("result", {}).get("results", [])
            except Exception as exc:
                print(f"  [warn] data.go.th query failed for {kw}: {exc}")
                continue

            q_count = 0
            for pkg in pkgs:
                if q_count >= per_query:
                    break
                if (pkg.get("license_id") or "") not in ALLOWED_LICENSES:
                    continue
                title = pkg.get("title") or ""
                if is_pii_blocked_title(title):
                    continue

                org = (pkg.get("organization") or {}).get("title") or "data.go.th"
                for res in pkg.get("resources", []):
                    if (res.get("format") or "").strip().upper() != "PDF":
                        continue
                    url = res.get("url") or ""
                    if not url or url in seen_urls:
                        continue
                    res_name = res.get("name") or ""
                    if is_pii_blocked_title(res_name):
                        continue

                    ok, sha256, size_bytes, num_pages, dest_path = download_and_validate_pdf(
                        session,
                        url,
                        dest_dir=SARABAN_RAW_DIR / saraban_type,
                        scanned_only=False,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if not ok or dest_path is None:
                        continue
                    if sha256 in seen_hashes:
                        dest_path.unlink(missing_ok=True)
                        continue

                    record = {
                        "doc_id": pkg["id"],
                        "resource_id": res["id"],
                        "saraban_type": saraban_type,
                        "saraban_subtype": subtype,
                        "meta_doc_no": "",
                        "meta_doc_date": (pkg.get("metadata_modified") or "")[:10],
                        "meta_subject": title,
                        "link_label": res_name,
                        "organization": org,
                        "source_portal": "data.go.th",
                        "source_url": url,
                        "local_path": str(dest_path.as_posix()),
                        "sha256": sha256,
                        "size_bytes": size_bytes,
                        "total_pages": num_pages,
                    }
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    mf.flush()

                    seen_urls.add(url)
                    seen_hashes.add(sha256)
                    type_counts[saraban_type] = type_counts.get(saraban_type, 0) + 1
                    added += 1
                    q_count += 1
                    print(
                        f"  + [data.go.th/{saraban_type}/{subtype}] "
                        f"{org[:30]} | {num_pages}p | {title[:40]}"
                    )
                    break

    print(f"\nExtra data.go.th collection added {added} PDFs. Updated totals: {type_counts}")


def _get_ckan_json(endpoint: str, headers: dict, params: dict | None = None) -> dict:
    resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"API error on {endpoint}: {data.get('error')}")
    return data["result"]


def run_eda() -> None:
    """Run Exploratory Data Analysis against Thailand Open Data (data.go.th) REST API."""
    api_key = load_datagoth_api_key()
    headers = {"api-key": api_key}

    print("=== 1. Portal Status & Entity Counts ===")
    status = _get_ckan_json("status_show", headers)
    print(f"Site Title   : {status.get('site_title')}")
    print(f"CKAN Version : {status.get('ckan_version')}")
    print(f"Site URL     : {status.get('site_url')}")

    groups = _get_ckan_json("group_list", headers, {"all_fields": True})
    orgs = _get_ckan_json("organization_list", headers)
    print(f"Groups       : {len(groups)}")
    print(f"Organizations: {len(orgs)}")

    print("\n=== 2. Dataset Facet Summary (package_search) ===")
    facet_fields = [
        "organization",
        "groups",
        "res_format",
        "license_id",
        "tags",
        "data_type",
        "update_frequency_unit",
    ]
    search_res = _get_ckan_json(
        "package_search",
        headers,
        {
            "rows": 0,
            "facet.field": json.dumps(facet_fields),
            "facet.limit": 15,
        },
    )
    print(f"Total Datasets: {search_res['count']:,}")

    for field, info in search_res.get("search_facets", {}).items():
        items = sorted(info.get("items", []), key=lambda x: x["count"], reverse=True)
        print(f"\n--- Top {field} ---")
        for item in items[:10]:
            print(f"  {item['display_name']} ({item['name']}): {item['count']:,}")

    print("\n=== 3. Latest 5 Modified Datasets ===")
    latest = _get_ckan_json("package_search", headers, {"rows": 5, "sort": "metadata_modified desc"})
    for idx, pkg in enumerate(latest["results"], 1):
        formats = sorted({r.get("format") or "N/A" for r in pkg.get("resources", [])})
        ds_active = any(r.get("datastore_active") for r in pkg.get("resources", []))
        print(
            f"{idx}. {pkg.get('title')} | org={pkg.get('organization', {}).get('title')} "
            f"| modified={pkg.get('metadata_modified', '')[:10]} "
            f"| resources={len(pkg.get('resources', []))} {formats} | datastore={ds_active}"
        )

    print("\n=== 4. Datastore API Sample Check ===")
    ds_candidates = _get_ckan_json(
        "package_search",
        headers,
        {"q": "res_format:CSV", "rows": 10, "sort": "metadata_modified desc"},
    )
    found_ds = None
    for pkg in ds_candidates["results"]:
        for res in pkg.get("resources", []):
            if res.get("datastore_active"):
                found_ds = (pkg.get("title"), res)
                break
        if found_ds:
            break

    if found_ds:
        pkg_title, res = found_ds
        print(f"Sample Datastore Resource: {pkg_title} -> {res.get('name')} ({res.get('id')})")
        ds_data = _get_ckan_json("datastore_search", headers, {"resource_id": res["id"], "limit": 3})
        print(f"Total rows in resource: {ds_data.get('total', 'N/A')}")
        print(f"Fields: {[f['id'] for f in ds_data.get('fields', [])]}")
        print(f"Sample record[0]: {ds_data.get('records', [None])[0]}")
    else:
        print("No datastore_active resource found in top 10 CSV results.")


def main_general(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect diverse Thai government PDFs from data.go.th.")
    parser.add_argument("--per-group", type=int, default=TARGET_PER_GROUP)
    parser.add_argument("--max-per-org", type=int, default=MAX_PER_ORG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scanned-only", action="store_true")
    parser.add_argument("--target-new", type=int, default=100)
    args = parser.parse_args(argv)
    collect_general(
        per_group=args.per_group,
        max_per_org=args.max_per_org,
        seed=args.seed,
        scanned_only=args.scanned_only,
        target_new=args.target_new,
    )


def main_saraban_extra(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect supplementary Saraban PDFs from data.go.th.")
    parser.add_argument("--per-query", type=int, default=10)
    args = parser.parse_args(argv)
    collect_saraban_extra(per_query=args.per_query)


def main_eda(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run EDA against data.go.th CKAN API.")
    parser.parse_args(argv)
    run_eda()


__all__ = [
    "DATA_GOTH_QUERIES",
    "collect_general",
    "collect_saraban_extra",
    "download_and_validate_general_pdf",
    "fetch_group_candidates",
    "has_scanned_pages",
    "load_existing_manifest",
    "main_eda",
    "main_general",
    "main_saraban_extra",
    "run_eda",
]
