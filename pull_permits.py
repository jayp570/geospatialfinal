"""
Step 1: Pull Austin issued construction permits (your development LABEL source).

Dataset: "Issued Construction Permits" on the City of Austin Open Data Portal
Socrata dataset ID: 3syk-w9eu
API docs:  https://dev.socrata.com/foundry/data.austintexas.gov/3syk-w9eu

Why start here (before parcels, before imagery): this dataset already carries a
`tcad_id` field, which is the exact join key to the Travis County TCAD parcel
layer. Confirming this join works is the single biggest risk in the whole
pipeline, so we de-risk it first with the cheapest possible pull.

Usage:
    python pull_permits.py --since 2021-01-01 --out permits_raw.csv
    (or the original: python pull_permits.py --years-back 5 --out permits_raw.csv)

This pulls ALL permit types (plumbing, electrical, irrigation, new
construction, remodel, etc.) for inspection. You will almost certainly NOT
want all of these as your "developed" label -- a plumbing permit on an
existing house is not new development. After running this once, open the
output and look at the distinct values of `permit_class_mapped` and
`work_class` to decide which rows count as "developed." A reasonable first
filter is permit_class_mapped == "Residential" or "Commercial" AND
work_class containing "New".
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.parse import urlencode
import json

SOCRATA_BASE = "https://data.austintexas.gov/resource/3syk-w9eu.json"
PAGE_SIZE = 5000


def fetch_page(where_clause: str, offset: int) -> list:
    params = {
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$where": where_clause,
        "$order": "issue_date",
    }
    url = f"{SOCRATA_BASE}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "ut-geoai-site-suitability/0.1"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years-back", type=int, default=5,
                     help="How many years of permit history to pull.")
    ap.add_argument("--since", default=None,
                     help="Pull permits issued on or after this date (YYYY-MM-DD). "
                          "Overrides --years-back. For the project's 2021-2025 label "
                          "window use --since 2021-01-01, so no part of 2021 is missed.")
    ap.add_argument("--out", default="permits_raw.csv")
    args = ap.parse_args()

    if args.since:
        cutoff = datetime.strptime(args.since, "%Y-%m-%d").strftime("%Y-%m-%dT00:00:00")
    else:
        cutoff = (datetime.utcnow() - timedelta(days=365 * args.years_back)).strftime("%Y-%m-%dT00:00:00")
    where_clause = f"issue_date >= '{cutoff}'"
    print(f"Pulling permits issued since {cutoff} ...")

    rows = []
    offset = 0
    while True:
        batch = fetch_page(where_clause, offset)
        if not batch:
            break
        rows.extend(batch)
        print(f"  fetched {len(rows)} rows so far (offset={offset})")
        offset += PAGE_SIZE
        if len(batch) < PAGE_SIZE:
            break
        time.sleep(0.2)  # be polite to the API

    if not rows:
        print("No rows returned -- check the $where clause / dataset ID.", file=sys.stderr)
        sys.exit(1)

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} permit records to {args.out}")
    print("\nNext: open this file and check distinct values of "
          "`permit_class_mapped` and `work_class` to decide your "
          "'developed' filter before running build_labels.py.")


if __name__ == "__main__":
    main()
