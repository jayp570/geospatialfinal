"""
Step 2: Pull TCAD parcel geometry + attributes from Travis County's GIS server.

Service: Travis County GIS, Boundaries_and_Jurisdictions/TCAD_public, layer 0
    https://gis.traviscountytx.gov/server1/rest/services/Boundaries_and_Jurisdictions/TCAD_public/MapServer/0

FIXED 2026-11-16: the address used to end in /FeatureServer/0. The county
serves this layer as a MapServer only, and the old address answered
"Server object extension 'featureserver' not found" (HTTP 500).

This is a standard Esri ArcGIS REST feature service, queried in pages via the
/query endpoint (resultOffset / resultRecordCount) since a single request
won't return every parcel in the county at once.

IMPORTANT: Before relying on this script, open the service's metadata page in
a browser --
    https://gis.traviscountytx.gov/server1/rest/services/Boundaries_and_Jurisdictions/TCAD_public/MapServer/0?f=pjson
-- and confirm: (1) the exact field name for the parcel ID that matches the
permit dataset's `tcad_id`, and (2) `maxRecordCount`, which controls how many
rows you get per page.

What the metadata page showed on 2026-11-16 (metadata only; no parcel rows
have been downloaded yet, so treat the id match as UNCONFIRMED):
    * maxRecordCount is 2000, so PAGE_SIZE = 1000 is fine.
    * Two id fields exist: PROP_ID (an integer) and geo_id (text).
      The permits' `tcad_id` values are 10-digit text like "0217650439",
      so `geo_id` is the likely match. Confirm it by comparing real values
      from the test pull against permits_raw.csv before the full run.
    * The layer is rebuilt monthly, so it shows TODAY's parcels, not 2020's,
      and it carries no appraisal values. See the README limitations.

Usage:
    python pull_parcels.py --out parcels_raw.geojson
"""
import argparse
import json
import time
from urllib.request import urlopen, Request
from urllib.parse import urlencode

SERVICE_URL = (
    "https://gis.traviscountytx.gov/server1/rest/services/"
    "Boundaries_and_Jurisdictions/TCAD_public/MapServer/0/query"
)
PAGE_SIZE = 1000  # drop this if the service's maxRecordCount is lower


def fetch_page(offset: int) -> dict:
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
    }
    url = f"{SERVICE_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "ut-geoai-site-suitability/0.1"})
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="parcels_raw.geojson")
    ap.add_argument("--max-pages", type=int, default=None,
                     help="Cap pages for a quick test pull before downloading the whole county.")
    args = ap.parse_args()

    all_features = []
    offset = 0
    page = 0
    while True:
        data = fetch_page(offset)
        feats = data.get("features", [])
        if not feats:
            break
        all_features.extend(feats)
        page += 1
        print(f"  page {page}: +{len(feats)} features (total {len(all_features)})")
        if len(feats) < PAGE_SIZE:
            break
        if args.max_pages and page >= args.max_pages:
            print("Hit --max-pages cap, stopping test pull.")
            break
        offset += PAGE_SIZE
        time.sleep(0.2)

    out = {"type": "FeatureCollection", "features": all_features}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Wrote {len(all_features)} parcels to {args.out}")
    if all_features:
        print("\nSample feature properties (inspect field names here):")
        print(json.dumps(all_features[0].get("properties", {}), indent=2)[:1000])


if __name__ == "__main__":
    main()
