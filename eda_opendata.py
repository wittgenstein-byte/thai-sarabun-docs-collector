"""Exploratory Data Analysis (EDA) for Thailand Open Data (data.go.th) REST API."""

import json
from pathlib import Path
import requests

BASE_URL = "https://opend.data.go.th/get-ckan"


def load_api_key(env_path: str = ".env") -> str:
    """Load API key from .env (supports either raw key or KEY=VALUE format)."""
    text = Path(env_path).read_text(encoding="utf-8").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
        return line
    raise ValueError("No API key found in .env")


def get_json(endpoint: str, headers: dict, params: dict | None = None) -> dict:
    resp = requests.get(
        f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"API error on {endpoint}: {data.get('error')}")
    return data["result"]


def main() -> None:
    api_key = load_api_key()
    headers = {"api-key": api_key}

    print("=== 1. Portal Status & Entity Counts ===")
    status = get_json("status_show", headers)
    print(f"Site Title   : {status.get('site_title')}")
    print(f"CKAN Version : {status.get('ckan_version')}")
    print(f"Site URL     : {status.get('site_url')}")

    groups = get_json("group_list", headers, {"all_fields": True})
    orgs = get_json("organization_list", headers)
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
    search_res = get_json(
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
    latest = get_json(
        "package_search",
        headers,
        {"rows": 5, "sort": "metadata_modified desc"},
    )
    for idx, pkg in enumerate(latest["results"], 1):
        formats = sorted({r.get("format") or "N/A" for r in pkg.get("resources", [])})
        ds_active = any(r.get("datastore_active") for r in pkg.get("resources", []))
        print(
            f"{idx}. {pkg.get('title')} | org={pkg.get('organization', {}).get('title')} "
            f"| modified={pkg.get('metadata_modified', '')[:10]} "
            f"| resources={len(pkg.get('resources', []))} {formats} | datastore={ds_active}"
        )

    print("\n=== 4. Datastore API Sample Check ===")
    # Search for a dataset with datastore_active resources
    ds_candidates = get_json(
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
        ds_data = get_json(
            "datastore_search",
            headers,
            {"resource_id": res["id"], "limit": 3},
        )
        print(f"Total rows in resource: {ds_data.get('total', 'N/A')}")
        print(f"Fields: {[f['id'] for f in ds_data.get('fields', [])]}")
        print(f"Sample record[0]: {ds_data.get('records', [None])[0]}")
    else:
        print("No datastore_active resource found in top 10 CSV results.")


if __name__ == "__main__":
    main()
