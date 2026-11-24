"""
Step 3: Join permits to parcels to build your labeled dataset.

This is the step that turns two raw downloads into the thing your model
actually trains on: one row per parcel, with a 0/1 "developed" label.

Run this AFTER:
    1. pull_permits.py  -> permits_raw.csv
    2. pull_parcels.py  -> parcels_raw.geojson
...and after you've inspected permits_raw.csv to decide which rows count as
real "development" (see the --developed-work-class-contains /
--developed-permit-classes flags below -- the defaults are a reasonable
starting guess, not gospel).

TWO WAYS TO MATCH PERMITS TO PARCELS (--label-mode)

  id        (default, the original behavior) Match the permit's tcad_id to the
            parcel's id field. Simple, but parcel ids CHANGE when a lot is
            subdivided or merged, so a 2023 permit can carry an id that did
            not exist in 2020. Those permits silently fail to match.

  location  (recommended for the final run) Drop each permit's point onto the
            parcel polygons and see which parcel it lands in (a "spatial
            join"). This works even when ids changed, as long as the parcel
            file is the 2020 snapshot. Needs latitude/longitude on the permits.

THE CUTOFF (--cutoff-year)

  The project predicts development AFTER 2020 from features dated 2020 or
  earlier. With --cutoff-year 2020 only permits issued 2021..--label-end-year
  count toward the label. Leave it off to keep the original behavior (every
  permit in the file counts).

  The parcel file should be the 2020 version, and you should restrict it to
  parcels that were vacant or underused in 2020 (using the 2020 appraisal
  roll) before or after this step. This script does not do that filtering.

STATUS: the id mode is the original starter code. The location mode and the
cutoff filter were added later and are syntax-checked but NOT yet run on real
data (no geopandas or real files were available where they were written).
Check the printed match counts carefully the first time.

Requires: pandas, geopandas (pip install pandas geopandas)

Usage (recommended final run):
    python build_labels.py \
        --permits permits_raw.csv \
        --parcels parcels_2020.geojson \
        --parcel-id-field TCAD_ID \
        --label-mode location --cutoff-year 2020 \
        --negative-sample-ratio 0 \
        --out labeled_parcels.geojson

Usage (original quick check):
    python build_labels.py --permits permits_raw.csv \
        --parcels parcels_raw.geojson --parcel-id-field TCAD_ID
"""
import argparse
import pandas as pd
import geopandas as gpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permits", default="permits_raw.csv")
    ap.add_argument("--parcels", default="parcels_raw.geojson")
    ap.add_argument("--parcel-id-field", default="TCAD_ID",
                     help="The field in the parcel layer that matches permits' tcad_id. "
                          "CONFIRM this against the sample properties pull_parcels.py printed.")
    ap.add_argument("--developed-permit-classes", nargs="+",
                     default=["Residential", "Commercial"])
    ap.add_argument("--developed-work-class-contains", default="New",
                     help="Substring match on work_class, e.g. 'New' catches "
                          "'New Construction'. Adjust after inspecting your data.")
    ap.add_argument("--negative-sample-ratio", type=float, default=1.0,
                     help="How many undeveloped parcels to sample per developed parcel. "
                          "Use 0 to KEEP ALL undeveloped parcels (recommended for the "
                          "final run, so the model is tested at the real base rate).")
    ap.add_argument("--label-mode", choices=["id", "location"], default="id",
                     help="How to match permits to parcels. See the top of this file.")
    ap.add_argument("--cutoff-year", type=int, default=None,
                     help="If set, only permits issued AFTER this year (through "
                          "--label-end-year) count. Use 2020 for the project design.")
    ap.add_argument("--label-end-year", type=int, default=2025)
    ap.add_argument("--date-col", default="issue_date")
    ap.add_argument("--permit-lat-col", default="latitude")
    ap.add_argument("--permit-lon-col", default="longitude")
    ap.add_argument("--out", default="labeled_parcels.geojson")
    args = ap.parse_args()

    permits = pd.read_csv(args.permits, dtype={"tcad_id": str}, low_memory=False)

    # ---- Which permits count as "development"? ------------------------------
    is_developed_permit = (
        permits["permit_class_mapped"].isin(args.developed_permit_classes)
        & permits["work_class"].str.contains(args.developed_work_class_contains, case=False, na=False)
    )
    if args.cutoff_year is not None:
        if args.date_col not in permits.columns:
            raise SystemExit(f"'{args.date_col}' not found in permits. Columns: {list(permits.columns)}")
        year = pd.to_datetime(permits[args.date_col], errors="coerce").dt.year
        in_window = (year > args.cutoff_year) & (year <= args.label_end_year)
        print(f"Label window: permits issued {args.cutoff_year + 1}-{args.label_end_year}. "
              f"{int((is_developed_permit & in_window).sum())} of "
              f"{int(is_developed_permit.sum())} qualifying permits fall inside it.")
        if (is_developed_permit & (year == args.cutoff_year + 1)).sum() == 0:
            print(f"WARNING: no qualifying permits from {args.cutoff_year + 1}. Did the permit "
                  "pull start late? Re-run pull_permits.py with --since "
                  f"{args.cutoff_year + 1}-01-01.")
        is_developed_permit &= in_window
    dev_permits = permits[is_developed_permit].copy()

    parcels = gpd.read_file(args.parcels)
    if args.parcel_id_field not in parcels.columns:
        raise SystemExit(
            f"'{args.parcel_id_field}' not found in parcel fields. Available fields:\n"
            f"{list(parcels.columns)}\nRe-run with --parcel-id-field set correctly."
        )

    if args.label_mode == "id":
        # ---- Original behavior: match on the parcel id ----------------------
        dev_ids = dev_permits["tcad_id"].str.strip().str.lstrip("0")
        developed_ids = set(dev_ids.dropna())
        print(f"Found {len(developed_ids)} unique parcel ids with a qualifying development permit.")
        parcels["_pid_norm"] = parcels[args.parcel_id_field].astype(str).str.strip().str.lstrip("0")
        parcels["developed"] = parcels["_pid_norm"].isin(developed_ids).astype(int)
        matched_ids = set(parcels.loc[parcels["developed"] == 1, "_pid_norm"])
        print(f"{len(matched_ids)} of those ids exist in the parcel file "
              f"({len(developed_ids) - len(matched_ids)} did not match; ids change when "
              "lots are subdivided, which is why --label-mode location exists).")
        parcels = parcels.drop(columns=["_pid_norm"])
    else:
        # ---- Location mode: which parcel polygon contains the permit point? -
        for col in (args.permit_lat_col, args.permit_lon_col):
            if col not in dev_permits.columns:
                raise SystemExit(f"'{col}' not found in permits. Columns: {list(permits.columns)}\n"
                                 "Set --permit-lat-col / --permit-lon-col.")
        lat = pd.to_numeric(dev_permits[args.permit_lat_col], errors="coerce")
        lon = pd.to_numeric(dev_permits[args.permit_lon_col], errors="coerce")
        has_xy = lat.notna() & lon.notna()
        print(f"{int(has_xy.sum())} of {len(dev_permits)} qualifying permits have coordinates.")
        # Permit coordinates are longitude/latitude (EPSG:4326). Convert them
        # to whatever coordinate system the parcels use before comparing.
        points = gpd.GeoDataFrame(
            dev_permits.loc[has_xy, []],
            geometry=gpd.points_from_xy(lon[has_xy], lat[has_xy]), crs="EPSG:4326",
        ).to_crs(parcels.crs)
        hits = gpd.sjoin(points, parcels[["geometry"]], how="inner", predicate="within")
        counts = hits["index_right"].value_counts()
        parcels["permit_count"] = parcels.index.map(counts).fillna(0).astype(int)
        parcels["developed"] = (parcels["permit_count"] > 0).astype(int)
        n_matched_permits = hits.index.nunique()
        print(f"{n_matched_permits} permits landed inside a parcel "
              f"({int(has_xy.sum()) - n_matched_permits} fell outside every parcel, e.g. in a "
              "road right-of-way or outside the parcel file's extent).")
        print("REMINDER: this is only correct if the parcel file is the 2020 snapshot.")

    # ---- Columns the later scripts expect -----------------------------------
    # parcel_id: a plain string id.  x, y: centroid in meters (UTM zone 14N),
    # used by train_models.py to build spatial cross-validation blocks.
    parcels["parcel_id"] = parcels[args.parcel_id_field].astype(str).str.strip()
    centroids = parcels.geometry.to_crs(epsg=32614).centroid
    parcels["x"], parcels["y"] = centroids.x, centroids.y

    developed = parcels[parcels["developed"] == 1]
    undeveloped_pool = parcels[parcels["developed"] == 0]
    if args.negative_sample_ratio > 0:
        n_negative = int(len(developed) * args.negative_sample_ratio)
        n_negative = min(n_negative, len(undeveloped_pool))
        undeveloped_sample = undeveloped_pool.sample(n=n_negative, random_state=42)
    else:
        undeveloped_sample = undeveloped_pool      # keep every undeveloped parcel

    labeled = pd.concat([developed, undeveloped_sample])
    labeled = gpd.GeoDataFrame(labeled, geometry="geometry", crs=parcels.crs)
    labeled.to_file(args.out, driver="GeoJSON")

    print(f"\nLabeled dataset: {len(developed)} developed + {len(undeveloped_sample)} undeveloped "
          f"= {len(labeled)} total rows -> {args.out}")
    print("\nIf developed count looks way too high or too low, revisit the "
          "--developed-permit-classes / --developed-work-class-contains filters "
          "and re-run.")


if __name__ == "__main__":
    main()
