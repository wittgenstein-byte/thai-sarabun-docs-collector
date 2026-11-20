"""Harvest diverse Thai government PDFs from data.go.th across all 15 groups."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pypdfium2 as pdfium
import requests

from config import (
    ALLOWED_LICENSES,
    BASE_URL,
    GROUPS,
    IMAGE_COVERAGE_THRESHOLD,
    MANIFEST_PATH,
    MAX_PDF_BYTES,
    MAX_PER_ORG,
    MIN_PDF_BYTES,
    RAW_PDF_DIR,
    REQUEST_TIMEOUT,
    TARGET_PER_GROUP,
    load_credentials,
)
from scan_pages import compute_image_coverage, pick_scan_indices


def load_existing_manifest(manifest_path: Path) -> tuple[set[str], set[str], dict[str, int], dict[str, int]]:
    """Load already collected resource IDs, SHA256 hashes, and per-group/org counts."""
    seen_resources: set[str] = set()
    seen_hashes: set[str] = set()
    group_counts: dict[str, int] = {g: 0 for g in GROUPS}
    org_counts: dict[str, int] = {}

    if not manifest_path.exists():
        return seen_resources, seen_hashes, group_counts, org_counts

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
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
    params = {
        "fq": f"res_format:PDF AND groups:{group}",
        "rows": rows,
        "start": start,
    }
    if query:
        params["q"] = query
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
        print(f"  [warn] API query failed ({group}, start={start}): {exc}")
        return []


def has_scanned_pages(doc: pdfium.PdfDocument, max_check: int = 10) -> tuple[int, int]:
    """Return (scanned_count, scanned_ocr_count) among sampled pages."""
    scanned = 0
    scanned_ocr = 0
    for idx in pick_scan_indices(len(doc), max_check):
        page = doc[idx]
        try:
            cov = compute_image_coverage(page)
            if cov > IMAGE_COVERAGE_THRESHOLD:
                tp = page.get_textpage()
                try:
                    chars = len((tp.get_text_range() or "").strip())
                finally:
                    tp.close()
                if chars < 20:
                    scanned += 1
                else:
                    scanned_ocr += 1
        finally:
            page.close()
    return scanned, scanned_ocr


def download_and_validate_pdf(
    session: requests.Session, url: str, dest_path: Path, scanned_only: bool = False
) -> tuple[bool, str, int, int]:
    """Download a PDF if valid and within size limits.

    When scanned_only=True, keeps only PDFs containing at least one scanned/scanned_ocr page.
    Returns (ok, sha256, size_bytes, total_pages).
    """
    try:
        head = session.head(url, allow_redirects=True, timeout=12)
        if head.status_code == 200:
            clen = head.headers.get("Content-Length")
            if clen and clen.isdigit():
                size_est = int(clen)
                if size_est > MAX_PDF_BYTES or size_est < MIN_PDF_BYTES:
                    return False, "", 0, 0
    except requests.RequestException:
        pass

    try:
        with session.get(url, stream=True, allow_redirects=True, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status_code != 200:
                return False, "", 0, 0

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    return False, "", 0, 0
                chunks.append(chunk)

            if total < MIN_PDF_BYTES:
                return False, "", 0, 0

            raw = b"".join(chunks)
            if not raw.lstrip().startswith(b"%PDF-"):
                return False, "", 0, 0

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(raw)

            doc = pdfium.PdfDocument(str(dest_path))
            try:
                num_pages = len(doc)
                if num_pages < 1:
                    dest_path.unlink(missing_ok=True)
                    return False, "", 0, 0
                if scanned_only:
                    sc, sc_ocr = has_scanned_pages(doc)
                    if sc + sc_ocr == 0:
                        dest_path.unlink(missing_ok=True)
                        return False, "", 0, 0
            finally:
                doc.close()

            sha256 = hashlib.sha256(raw).hexdigest()
            return True, sha256, total, num_pages
    except Exception:
        dest_path.unlink(missing_ok=True)
        return False, "", 0, 0


def collect(
    per_group: int,
    max_per_org: int,
    seed: int,
    scanned_only: bool = False,
    target_new: int = 100,
) -> None:
    creds = load_credentials()
    rng = random.Random(seed)

    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_resources, seen_hashes, group_counts, org_counts = load_existing_manifest(MANIFEST_PATH)

    session = requests.Session()
    session.headers.update({"User-Agent": "thai-govern-ocr-benchmark/1.0"})

    added_new = 0
    start_offsets = (0, 120, 240, 360) if scanned_only else (0, 120)

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
                batch = fetch_group_candidates(
                    session, creds.datagoth_api_key, group, rows=120, start=start
                )
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

                org_obj = pkg.get("organization") or {}
                org_id = org_obj.get("name") or "unknown"
                org_title = org_obj.get("title") or org_id
                if org_counts.get(org_id, 0) >= max_per_org:
                    continue

                pdf_resources = [
                    r for r in pkg.get("resources", [])
                    if (r.get("format") or "").strip().upper() == "PDF" and r.get("url")
                ]
                if not pdf_resources:
                    continue

                res = rng.choice(pdf_resources)
                res_id = res["id"]
                if res_id in seen_resources:
                    continue

                dest_path = RAW_PDF_DIR / group / f"{res_id}.pdf"
                ok, sha256, size_bytes, total_pages = download_and_validate_pdf(
                    session, res["url"], dest_path, scanned_only=scanned_only
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
    print(f"\nCollection complete: added {added_new} new PDFs (total {total_docs} PDFs across {len(org_counts)} orgs).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect diverse Thai government PDFs.")
    parser.add_argument("--per-group", type=int, default=TARGET_PER_GROUP)
    parser.add_argument("--max-per-org", type=int, default=MAX_PER_ORG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scanned-only", action="store_true")
    parser.add_argument("--target-new", type=int, default=100)
    args = parser.parse_args()
    collect(
        per_group=args.per_group,
        max_per_org=args.max_per_org,
        seed=args.seed,
        scanned_only=args.scanned_only,
        target_new=args.target_new,
    )


if __name__ == "__main__":
    main()
