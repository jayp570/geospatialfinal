"""
Step 5: build features.csv, the one table train_models.py reads.

AI NOTE: written with Claude (Anthropic). Tested on small fake geometries
only (tests/test_build_features.py). NOT yet run on real Austin data.

What it does
    Starts from labeled_parcels.geojson (from build_labels.py: one row per
    2020 parcel with parcel_id, developed, x, y and the parcel shape) and adds
    the tabular columns in the README schema:

    lot_area                 square meters, measured from the parcel shape
    improvement_value_ratio  improvement value / (land + improvement value),
                             from the 2020 appraisal roll
    zoning_class             base zoning district at the parcel's interior point
    in_floodplain            1 if the flood layer covers more than
                             --flood-min-share of the parcel
    dist_to_road             meters from the parcel edge to the nearest road line
    slope, elevation,        copied from the terrain CSV written by
    land_cover_class         ee_extract_embeddings.py --terrain-out

    It then keeps only the parcels that were vacant or underused in 2020
    (improvement_value_ratio <= --max-improvement-ratio), optionally keeps only
    parcels inside the Austin city limits, joins the imagery embeddings, and
    writes features.csv.

    Every layer is optional. A column whose layer you did not give is left out,
    and the script prints the exact --numeric-cols / --categorical-cols to pass
    to train_models.py so it does not ask for the missing ones.

Background for non-geographers
    * All distances and areas are computed in EPSG:32614 (UTM zone 14N), a
      coordinate system measured in meters that is accurate around Austin.
      Longitude/latitude is in degrees, and a "degree" is not a fixed length.
    * "Interior point" (shapely's representative_point) is a point guaranteed
      to lie inside the parcel. A centroid can fall outside an L-shaped lot.
    * improvement_value_ratio near 0 means the land is worth far more than
      what is built on it: a vacant or underused lot.

THE CUTOFF
    The label is "got a new-construction permit in 2021-2025", so every
    feature must describe 2020 or earlier. This script cannot see what year a
    file describes. You tell it with --appraisal-year and --parcel-year, and
    it refuses years after --cutoff-year. Zoning is usually available only as
    it is today; if you use it, say so as a limitation (see the README).

Requires: pandas, geopandas

Usage (every layer)
    python build_features.py \
        --labels labeled_parcels.geojson --parcel-year 2020 \
        --appraisal appraisal_2020.csv --appraisal-year 2020 \
        --appraisal-id-col prop_id --labels-join-col PROP_ID \
        --land-value-col land_value --improvement-value-col improvement_value \
        --zoning zoning.geojson --zoning-field ZONING_ZTYPE \
        --floodplain flood_hazard.geojson \
        --roads tl_2020_48453_roads.shp \
        --city-limits austin_full_purpose.geojson \
        --terrain terrain.csv --embeddings embeddings.csv \
        --out features.csv
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd

WORK_CRS = "EPSG:32614"   # UTM zone 14N, meters

# README schema, in the order train_models.py lists them.
NUMERIC_SCHEMA = ["lot_area", "in_floodplain", "slope", "elevation",
                  "dist_to_road", "improvement_value_ratio"]
CATEGORICAL_SCHEMA = ["zoning_class", "land_cover_class"]


# -----------------------------------------------------------------------------
# Pure helpers (no files needed, so they can be unit-tested)
# -----------------------------------------------------------------------------
def base_zoning(value):
    """Collapse a full Austin zoning string to its base district.

    Austin zoning strings stack suffixes onto a base district:
    "SF-3-NP" is base SF-3 plus a neighborhood plan, "CS-MU-V-CO-NP" is base
    CS plus four overlays. Left alone there are hundreds of rare strings.
    Rule: drop a leading "I-" (interim zoning), then keep the first letters
    and, if the next piece starts with a digit, that piece too.

        SF-3-NP -> SF-3      CS-MU-V-CO-NP -> CS     MF-4-CO -> MF-4
        SF-4A   -> SF-4A     I-SF-2        -> SF-2   PUD     -> PUD
    Empty or unreadable values become "missing".
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "missing"
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return "missing"
    text = re.sub(r"^I-", "", text)
    match = re.match(r"^([A-Z]+)(-\d+[A-Z]?)?", text)
    return match.group(0) if match else "missing"


def improvement_ratio(land_value, improvement_value):
    """improvement / (land + improvement). NaN when the total is 0 or missing."""
    land = pd.to_numeric(pd.Series(land_value), errors="coerce").to_numpy(float)
    imp = pd.to_numeric(pd.Series(improvement_value), errors="coerce").to_numpy(float)
    total = land + imp
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, imp / total, np.nan)
    return ratio


def norm_id(series):
    """Ids as trimmed text with leading zeros removed, so "0012" matches "12"
    and the integer 12. An all-zero id becomes "0"."""
    s = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.str.lstrip("0").where(lambda v: v != "", "0")


def check_year(name, year, cutoff):
    if year is not None and year > cutoff:
        sys.exit(f"ERROR: {name} is {year}, which is after the cutoff year {cutoff}. "
                 "A feature from after the cutoff can show the outcome being predicted. "
                 "Use an earlier file, or change --cutoff-year only if the whole design changes.")


# -----------------------------------------------------------------------------
# One function per feature. Each takes parcels in WORK_CRS and returns a Series
# aligned to parcels.index.
# -----------------------------------------------------------------------------
def lot_area(parcels):
    return parcels.geometry.area


def interior_points(parcels):
    import geopandas as gpd
    return gpd.GeoDataFrame(geometry=parcels.geometry.representative_point(),
                            index=parcels.index, crs=parcels.crs)


def zoning_class(parcels, zoning, field):
    """Base zoning district of the zoning polygon under each parcel's interior
    point. Parcels under no polygon get "missing"."""
    import geopandas as gpd
    if field not in zoning.columns:
        sys.exit(f"ERROR: zoning field '{field}' not found. Columns: {list(zoning.columns)}")
    z = zoning[[field, "geometry"]].to_crs(parcels.crs)
    hit = gpd.sjoin(interior_points(parcels), z, how="left", predicate="within")
    first = hit[~hit.index.duplicated(keep="first")][field]   # overlapping polygons: take one
    return first.reindex(parcels.index).map(base_zoning)


def floodplain_share(parcels, flood):
    """Share of each parcel's area covered by the flood polygons (0 to 1)."""
    import geopandas as gpd
    f = flood[["geometry"]].to_crs(parcels.crs)
    f = f[f.geometry.notna() & ~f.geometry.is_empty]
    pairs = gpd.sjoin(parcels[["geometry"]], f, how="inner", predicate="intersects")
    share = pd.Series(0.0, index=parcels.index)
    if len(pairs):
        flood_geom = f.geometry.loc[pairs["index_right"]].to_numpy()
        parcel_geom = parcels.geometry.loc[pairs.index]
        # Union the flood pieces per parcel first, so overlapping flood
        # polygons are not counted twice.
        pieces = gpd.GeoSeries(flood_geom, index=pairs.index, crs=parcels.crs)
        merged = pieces.groupby(level=0).agg(lambda g: g.union_all())
        merged = gpd.GeoSeries(merged, crs=parcels.crs)
        own = parcels.geometry.loc[merged.index]
        covered = own.intersection(merged, align=True).area
        share.loc[merged.index] = (covered / own.area).clip(0, 1).to_numpy()
    return share


def dist_to_road(parcels, roads):
    """Meters from the parcel edge to the nearest road line. 0 if a road
    crosses the parcel."""
    import geopandas as gpd
    r = roads[["geometry"]].to_crs(parcels.crs)
    r = r[r.geometry.notna() & ~r.geometry.is_empty]
    near = gpd.sjoin_nearest(parcels[["geometry"]], r, how="left", distance_col="_d")
    return near.groupby(level=0)["_d"].min().reindex(parcels.index)


def inside_limits(parcels, limits):
    """True where the parcel's interior point lies inside the city-limit shape."""
    import geopandas as gpd
    lim = limits[["geometry"]].to_crs(parcels.crs)
    hit = gpd.sjoin(interior_points(parcels), lim, how="inner", predicate="within")
    return pd.Series(parcels.index.isin(hit.index), index=parcels.index)


def appraisal_ratio(appraisal, id_col, land_col, imp_col):
    """One improvement_value_ratio per property id.

    An appraisal roll can hold several rows for one property (for example one
    per owner of an undivided interest). Values are summed per id before the
    ratio is taken. The ratio is the same whether those rows split the value
    or repeat it.
    """
    for c in (id_col, land_col, imp_col):
        if c not in appraisal.columns:
            sys.exit(f"ERROR: appraisal column '{c}' not found. Columns: {list(appraisal.columns)}")
    a = pd.DataFrame({
        "_key": norm_id(appraisal[id_col]),
        "land": pd.to_numeric(appraisal[land_col], errors="coerce"),
        "imp": pd.to_numeric(appraisal[imp_col], errors="coerce"),
    })
    n_dup = int(a["_key"].duplicated().sum())
    if n_dup:
        print(f"  note: {n_dup:,} extra appraisal rows share an id with another row; summed per id.")
    g = a.groupby("_key")[["land", "imp"]].sum(min_count=1)
    return pd.Series(improvement_ratio(g["land"], g["imp"]), index=g.index)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build features.csv for train_models.py.")
    ap.add_argument("--labels", required=True, help="labeled_parcels.geojson from build_labels.py.")
    ap.add_argument("--parcel-year", type=int, default=None,
                    help="Year the parcel shapes describe. Refused if after --cutoff-year.")
    ap.add_argument("--cutoff-year", type=int, default=2020)
    ap.add_argument("--allow-current-parcels", action="store_true",
                    help="Skip the --parcel-year check. Only for a dry run with today's parcels; "
                         "say so as a limitation if you report results from it.")
    # appraisal roll
    ap.add_argument("--appraisal", help="CSV from the appraisal roll: id, land value, improvement value.")
    ap.add_argument("--appraisal-year", type=int, default=None,
                    help="Year of the appraisal roll. Required with --appraisal.")
    ap.add_argument("--appraisal-id-col", default="prop_id")
    ap.add_argument("--labels-join-col", default="parcel_id",
                    help="Column in the labels file holding the same id as --appraisal-id-col.")
    ap.add_argument("--land-value-col", default="land_value")
    ap.add_argument("--improvement-value-col", default="improvement_value")
    ap.add_argument("--max-improvement-ratio", type=float, default=0.25,
                    help="Vacant/underused filter: keep parcels with improvement_value_ratio at or "
                         "below this. 1.0 keeps every parcel. THE TEAM MUST CHOOSE AND REPORT THIS.")
    # layers
    ap.add_argument("--zoning", help="Zoning polygons.")
    ap.add_argument("--zoning-field", default="ZONING_ZTYPE")
    ap.add_argument("--floodplain", help="Flood hazard polygons (only the zones that count).")
    ap.add_argument("--flood-min-share", type=float, default=0.01,
                    help="in_floodplain = 1 when more than this share of the parcel is covered. "
                         "A small positive value ignores slivers from misaligned boundaries.")
    ap.add_argument("--roads", help="Road centerlines.")
    ap.add_argument("--city-limits", help="Polygon(s) to keep parcels inside (for example Austin full purpose).")
    ap.add_argument("--terrain", help="terrain.csv from ee_extract_embeddings.py --terrain-out.")
    ap.add_argument("--embeddings", help="embeddings.csv from ee_extract_embeddings.py.")
    ap.add_argument("--out", default="features.csv")
    ap.add_argument("--tabular-out", default=None, help="Also write the tabular columns alone.")
    args = ap.parse_args()

    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("This script needs geopandas:  pip install geopandas")

    check_year("--appraisal-year", args.appraisal_year, args.cutoff_year)
    if not args.allow_current_parcels:
        if args.parcel_year is None:
            sys.exit("ERROR: pass --parcel-year (the year the parcel file describes), or "
                     "--allow-current-parcels for a dry run with today's parcels.")
        check_year("--parcel-year", args.parcel_year, args.cutoff_year)
    if args.appraisal and args.appraisal_year is None:
        sys.exit("ERROR: --appraisal needs --appraisal-year so the cutoff can be checked.")

    parcels = gpd.read_file(args.labels)
    for c in ("parcel_id", "developed"):
        if c not in parcels.columns:
            sys.exit(f"ERROR: '{c}' not found in {args.labels}. Run build_labels.py first.")
    parcels["parcel_id"] = parcels["parcel_id"].astype(str).str.strip()
    parcels = parcels[parcels.geometry.notna() & ~parcels.geometry.is_empty]
    n0 = len(parcels)
    parcels = parcels.drop_duplicates(subset="parcel_id", keep="first").reset_index(drop=True)
    print(f"Read {n0:,} labeled parcels ({n0 - len(parcels):,} duplicate ids dropped), "
          f"{int(parcels['developed'].sum()):,} developed.")
    parcels = parcels.to_crs(WORK_CRS)
    parcels["geometry"] = parcels.geometry.buffer(0)       # repairs minor invalid shapes

    out = pd.DataFrame({"parcel_id": parcels["parcel_id"], "developed": parcels["developed"].astype(int)})
    centroid = parcels.geometry.centroid
    out["x"], out["y"] = centroid.x, centroid.y            # meters, EPSG:32614
    out["lot_area"] = lot_area(parcels)
    keep = pd.Series(True, index=parcels.index)

    if args.city_limits:
        inside = inside_limits(parcels, gpd.read_file(args.city_limits))
        print(f"City limits: {int(inside.sum()):,} of {len(inside):,} parcels are inside.")
        keep &= inside

    if args.appraisal:
        if args.labels_join_col not in parcels.columns:
            sys.exit(f"ERROR: --labels-join-col '{args.labels_join_col}' not in the labels file. "
                     f"Columns: {list(parcels.columns)}")
        ratio = appraisal_ratio(pd.read_csv(args.appraisal, dtype=str, low_memory=False),
                                args.appraisal_id_col, args.land_value_col, args.improvement_value_col)
        out["improvement_value_ratio"] = norm_id(parcels[args.labels_join_col]).map(ratio).to_numpy()
        matched = out["improvement_value_ratio"].notna()
        print(f"Appraisal {args.appraisal_year}: {int(matched.sum()):,} of {len(out):,} parcels "
              f"matched a roll record with a value ({matched.mean():.1%}). "
              "A low match rate usually means the wrong id column.")
        if args.max_improvement_ratio < 1.0:
            pool = out["improvement_value_ratio"] <= args.max_improvement_ratio
            print(f"Vacant/underused filter (ratio <= {args.max_improvement_ratio}): "
                  f"{int(pool.sum()):,} parcels pass; {int((matched & ~pool).sum()):,} are built up; "
                  f"{int((~matched).sum()):,} have no value and are dropped.")
            keep &= pool
    elif args.max_improvement_ratio < 1.0:
        print("WARNING: no --appraisal given, so the vacant/underused filter was NOT applied "
              "and improvement_value_ratio is absent. The design calls for both.")

    if args.zoning:
        out["zoning_class"] = zoning_class(parcels, gpd.read_file(args.zoning), args.zoning_field).to_numpy()
        print(f"Zoning: {int((out['zoning_class'] != 'missing').sum()):,} parcels got a district; "
              f"{out['zoning_class'].nunique()} distinct base districts.")
    if args.floodplain:
        share = floodplain_share(parcels, gpd.read_file(args.floodplain))
        out["in_floodplain"] = (share > args.flood_min_share).astype(int).to_numpy()
        print(f"Floodplain: {int(out['in_floodplain'].sum()):,} parcels flagged.")
    if args.roads:
        out["dist_to_road"] = dist_to_road(parcels, gpd.read_file(args.roads)).to_numpy()
        print(f"Roads: median distance {out['dist_to_road'].median():.1f} m.")

    if args.terrain:
        terrain = pd.read_csv(args.terrain, dtype={"parcel_id": str})
        cols = [c for c in ("slope", "elevation", "land_cover_class") if c in terrain.columns]
        terrain = terrain.drop_duplicates("parcel_id")[["parcel_id"] + cols]
        out = out.merge(terrain, on="parcel_id", how="left")
        print(f"Terrain: {int(out[cols[0]].notna().sum()) if cols else 0:,} parcels matched; columns {cols}.")

    out = out[keep.to_numpy()].reset_index(drop=True)
    if out.empty:
        sys.exit("ERROR: no parcels left after filtering. Check the id columns and the filters.")
    if out["developed"].nunique() < 2:
        sys.exit("ERROR: after filtering, every parcel has the same label. Check the filters.")

    numeric = [c for c in NUMERIC_SCHEMA if c in out.columns]
    categorical = [c for c in CATEGORICAL_SCHEMA if c in out.columns]
    out = out[["parcel_id", "developed", "x", "y"] + numeric + categorical]
    if args.tabular_out:
        out.to_csv(args.tabular_out, index=False)

    if args.embeddings:
        emb = pd.read_csv(args.embeddings, dtype={"parcel_id": str}).drop_duplicates("parcel_id")
        emb = emb.drop(columns=[c for c in emb.columns if c in out.columns and c != "parcel_id"])
        out = out.merge(emb, on="parcel_id", how="left")
        probe = next((c for c in out.columns if c.startswith("emb_A")), None)
        if probe:
            print(f"Embeddings: {int(out[probe].notna().sum()):,} of {len(out):,} parcels have imagery features.")
    else:
        print("NOTE: no --embeddings given. train_models.py needs the emb_A* columns.")

    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out):,} parcels x {out.shape[1]} columns to {args.out}. "
          f"Developed: {int(out['developed'].sum()):,} ({out['developed'].mean():.1%}).")
    missing = [c for c in NUMERIC_SCHEMA + CATEGORICAL_SCHEMA if c not in out.columns]
    if missing:
        print(f"Columns in the README schema that are absent: {missing}")
        print("Run train_models.py with:")
        print(f"  --numeric-cols {' '.join(numeric)} --categorical-cols {' '.join(categorical)}")


if __name__ == "__main__":
    main()
