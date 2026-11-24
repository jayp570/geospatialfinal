"""
ee_extract_embeddings.py -- per-parcel imagery features from Google Earth Engine.

STATUS: written and syntax-checked, but NOT yet run against Earth Engine
(the environment it was written in had no Earth Engine access). Run the
--pilot first and look at the output before trusting a full run.

What this produces
    One CSV row per parcel with the imagery features train_models.py expects:

    emb_A00..emb_A63   mean of the 64 embedding bands INSIDE the parcel, 2020
    nbr_A00..nbr_A63   mean in a 250 m RING around the parcel (the parcel
                       itself is cut out, so this describes the surroundings)
    emb_change         how much the parcel changed 2018 -> 2020
    nbr_change         how much the surroundings changed 2018 -> 2020
    is_tiny_parcel     1 if the parcel is smaller than about 3 pixels

    With --terrain-out terrain.csv it ALSO writes three tabular features per
    parcel (added later; same NOT-YET-RUN status as the rest of this file):

    elevation          mean ground elevation in meters (USGS 3DEP, 10 m)
    slope              mean slope in PERCENT (rise / run x 100), from 3DEP
    land_cover_class   most common NLCD 2019 class inside the parcel
    nlcd_code          the NLCD number behind that class name

    These are tabular features for the parcel-only model, not imagery
    features. They are in a separate file so train_models.py never mistakes
    them for embeddings. build_features.py --terrain reads the file.

Background for non-geographers
    * The dataset is Google's Satellite Embedding V1 (AlphaEarth Foundations).
      A foundation model has already turned a year of satellite imagery into
      64 numbers for every 10 m x 10 m pixel on land. Each pixel's 64 numbers
      form a unit-length vector that summarizes what that spot looked like
      over the year. We do not run any neural network ourselves; we only
      average those ready-made numbers over each parcel's shape.
    * This replaces the Clay image-chip approach from the proposal. Clay
      needs a fixed square image (~150 m across), which does not match the
      shape of a parcel. Per-pixel embeddings can be averaged over the
      parcel's real boundary.
    * "Change" is 1 minus the cosine similarity between the 2018 and 2020
      mean vectors: 0 means the place looked the same, larger means it
      changed (clearing, grading, new rooftops).
    * LEAKAGE RULE: both years must be 2020 or earlier, because the label is
      "got a permit in 2021-2025". An embedding from 2021+ could show the new
      building itself. The script refuses years after --cutoff-year.

One-time setup
    pip install earthengine-api geopandas
    earthengine authenticate            # opens a browser login
    You also need a Google Cloud project registered for Earth Engine
    (free for education/research): https://earthengine.google.com/

Usage
    # 1. Pilot: 50 parcels, to check everything works
    python ee_extract_embeddings.py --project YOUR_PROJECT_ID \
        --parcels labeled_parcels.geojson --out embeddings_pilot.csv --pilot 50

    # 2. Full run (safe to stop and restart; finished parcels are skipped)
    python ee_extract_embeddings.py --project YOUR_PROJECT_ID \
        --parcels labeled_parcels.geojson --out embeddings.csv \
        --terrain-out terrain.csv

    # Terrain and land cover only (skips the embeddings)
    python ee_extract_embeddings.py --project YOUR_PROJECT_ID \
        --parcels labeled_parcels.geojson --terrain-out terrain.csv --terrain-only

Attribution (required by the CC-BY 4.0 license; put it in your slides/report)
    "The AlphaEarth Foundations Satellite Embedding dataset is produced by
    Google and Google DeepMind."
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
WORK_CRS = "EPSG:32614"     # UTM zone 14N: meters, correct for Austin
PIXEL_M = 10                # native pixel size of the embeddings
TINY_AREA_M2 = 300          # ~3 pixels; smaller parcels use the fallback below
TINY_BUFFER_M = 15          # tiny parcel -> circle of this radius at its centroid
RING_M = 250                # neighborhood ring width

OUT_COLUMNS = (["parcel_id", "is_tiny_parcel", "parcel_area_m2"]
               + [f"emb_{b}" for b in BANDS] + [f"nbr_{b}" for b in BANDS]
               + ["emb_change", "nbr_change"])

# ---- Terrain and land cover (--terrain-out) ---------------------------------
ELEVATION_ASSET = "USGS/3DEP/10m"                    # ee.Image, band "elevation", meters
NLCD_COLLECTION = "USGS/NLCD_RELEASES/2019_REL/NLCD"  # ee.ImageCollection, band "landcover"
NLCD_YEAR = 2019            # must be <= the cutoff year; main() checks this
TERRAIN_COLUMNS = ["parcel_id", "elevation", "slope", "nlcd_code", "land_cover_class"]
# NLCD legend (class number -> short name). 30 m pixels.
NLCD_CLASSES = {
    11: "open_water", 12: "perennial_ice_snow",
    21: "developed_open_space", 22: "developed_low", 23: "developed_medium",
    24: "developed_high", 31: "barren", 41: "deciduous_forest",
    42: "evergreen_forest", 43: "mixed_forest", 52: "shrub_scrub",
    71: "grassland", 81: "pasture_hay", 82: "cultivated_crops",
    90: "woody_wetlands", 95: "emergent_wetlands",
}


# -----------------------------------------------------------------------------
# Pure-numpy helpers (no Earth Engine needed, so these can be unit-tested)
# -----------------------------------------------------------------------------
def vector(props, prefix):
    """Pull 64 band means out of an Earth Engine property dict.
    Missing values (parcel fell on masked pixels) become NaN."""
    return np.array([np.nan if props.get(f"{prefix}_{b}") is None
                     else float(props[f"{prefix}_{b}"]) for b in BANDS])


def cosine_change(v_then, v_now):
    """1 - cosine similarity. 0 = unchanged; larger = more change."""
    if np.isnan(v_then).any() or np.isnan(v_now).any():
        return np.nan
    denom = np.linalg.norm(v_then) * np.linalg.norm(v_now)
    return np.nan if denom == 0 else float(1 - np.dot(v_then, v_now) / denom)


def assemble_row(parcel_id, is_tiny, area_m2, parcel_props, ring_props):
    """Turn the raw Earth Engine results for one parcel into one CSV row."""
    emb_now, emb_then = vector(parcel_props, "now"), vector(parcel_props, "then")
    nbr_now, nbr_then = vector(ring_props, "now"), vector(ring_props, "then")
    row = {"parcel_id": parcel_id, "is_tiny_parcel": int(is_tiny),
           "parcel_area_m2": round(float(area_m2), 1)}
    row.update({f"emb_{b}": v for b, v in zip(BANDS, emb_now)})
    row.update({f"nbr_{b}": v for b, v in zip(BANDS, nbr_now)})
    row["emb_change"] = cosine_change(emb_then, emb_now)
    row["nbr_change"] = cosine_change(nbr_then, nbr_now)
    return row


def nlcd_class_name(code):
    """NLCD class number -> short name. Missing -> "missing"; a number that is
    not in the legend -> "nlcd_<number>" so it is visible, not hidden."""
    if code is None:
        return "missing"
    try:
        if np.isnan(float(code)):
            return "missing"
        code = int(round(float(code)))
    except (TypeError, ValueError):
        return "missing"
    return NLCD_CLASSES.get(code, f"nlcd_{code}")


def assemble_terrain_row(parcel_id, mean_props, cover_props):
    """Turn the raw Earth Engine results for one parcel into one terrain row.

    mean_props holds the mean "elevation" and "slope" inside the parcel.
    cover_props holds the most common land-cover number. Earth Engine names a
    single-band result after the reducer ("mode"), so both names are accepted.
    """
    def number(props, key):
        v = props.get(key)
        return np.nan if v is None else float(v)

    code = cover_props.get("landcover", cover_props.get("mode"))
    return {"parcel_id": parcel_id,
            "elevation": round(number(mean_props, "elevation"), 2),
            "slope": round(number(mean_props, "slope"), 3),
            "nlcd_code": "" if nlcd_class_name(code) == "missing" else int(round(float(code))),
            "land_cover_class": nlcd_class_name(code)}


# -----------------------------------------------------------------------------
# Geometry preparation (runs on your computer, before anything is uploaded)
# -----------------------------------------------------------------------------
def prepare_parcels(path, id_field, pilot):
    """Read parcels and build the two shapes we average over for each one."""
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("This script needs geopandas:  pip install geopandas")

    gdf = gpd.read_file(path)
    if id_field not in gdf.columns:
        sys.exit(f"ERROR: id field '{id_field}' not found. Columns: {list(gdf.columns)}\n"
                 "Pass --id-field with the right name.")
    gdf = gdf[[id_field, "geometry"]].rename(columns={id_field: "parcel_id"})
    gdf["parcel_id"] = gdf["parcel_id"].astype(str)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf.drop_duplicates(subset="parcel_id")
    if pilot:
        gdf = gdf.head(pilot)

    # Work in meters. Areas and buffers are meaningless in degrees.
    gdf = gdf.to_crs(WORK_CRS)
    gdf["geometry"] = gdf.geometry.buffer(0)        # repairs minor invalid shapes
    gdf["area_m2"] = gdf.geometry.area
    gdf["is_tiny"] = gdf["area_m2"] < TINY_AREA_M2

    # Shape 1: where we average the parcel's own embedding.
    # A parcel smaller than ~3 pixels has too few pixels for a stable average,
    # so we use a small circle at its centroid instead (about 7 pixels).
    # LIMITATION: that circle spills onto the neighbors.
    sample = gdf.geometry.copy()
    sample[gdf["is_tiny"]] = gdf.geometry[gdf["is_tiny"]].centroid.buffer(TINY_BUFFER_M)

    # Shape 2: the 250 m ring. Buffer outward, then cut the parcel out.
    # (quad_segs=4 keeps the circle arcs coarse so uploads stay small.)
    ring = gdf.geometry.buffer(RING_M, quad_segs=4).difference(gdf.geometry)

    # Earth Engine wants longitude/latitude.
    gdf["sample_ll"] = gpd.GeoSeries(sample, crs=WORK_CRS).to_crs(4326)
    gdf["ring_ll"] = gpd.GeoSeries(ring, crs=WORK_CRS).to_crs(4326)
    bounds = gpd.GeoSeries(gdf.geometry.buffer(RING_M + 100), crs=WORK_CRS).to_crs(4326).total_bounds
    return gdf, [float(b) for b in bounds]


def to_ee_geometry(ee, shape):
    from shapely.geometry import mapping
    # json round-trip turns shapely's tuples into plain lists, which is the
    # form Earth Engine accepts.
    return ee.Geometry(json.loads(json.dumps(mapping(shape))))


# -----------------------------------------------------------------------------
# Earth Engine
# -----------------------------------------------------------------------------
def build_image(ee, year_now, year_then, bounds):
    """One 128-band image: 64 bands for each of the two years."""
    region = ee.Geometry.Rectangle(bounds)
    collection = ee.ImageCollection(COLLECTION)

    def one_year(year, prefix):
        start = ee.Date.fromYMD(year, 1, 1)
        # The dataset is stored as many tiles; mosaic() stitches the tiles
        # that cover Austin into a single image for that year.
        image = (collection.filterDate(start, start.advance(1, "year"))
                 .filterBounds(region).mosaic().select(BANDS))
        return image.rename([f"{prefix}_{b}" for b in BANDS])

    return one_year(year_now, "now").addBands(one_year(year_then, "then"))


def build_terrain_images(ee):
    """Two images: (elevation + slope) to average, and land cover to take the
    most common class of."""
    elevation = ee.Image(ELEVATION_ASSET).select("elevation")
    # ee.Terrain.slope returns DEGREES. The project schema wants PERCENT:
    # percent = tan(angle) x 100.  (45 degrees = 100 percent.)
    slope_pct = (ee.Terrain.slope(elevation).multiply(np.pi / 180).tan()
                 .multiply(100).rename("slope"))
    land_cover = (ee.ImageCollection(NLCD_COLLECTION)
                  .filter(ee.Filter.eq("system:index", str(NLCD_YEAR)))
                  .first().select("landcover"))
    return elevation.addBands(slope_pct), land_cover


def process_terrain_batch(ee, images, batch, tile_scale, depth=0):
    """Terrain version of process_batch: same split-in-half retry."""
    terrain_image, cover_image = images
    ids = batch["parcel_id"].tolist()
    try:
        means = reduce_shapes(ee, terrain_image, ids, batch["sample_ll"], tile_scale)
        # mode = the most common value. Land-cover classes are labels, so an
        # average of them would be meaningless.
        cover = reduce_shapes(ee, cover_image, ids, batch["sample_ll"], tile_scale,
                              reducer=ee.Reducer.mode())
    except Exception as err:  # ee.EEException and network errors
        if len(batch) == 1:
            print(f"    WARNING: parcel {ids[0]} failed ({str(err)[:120]}); skipped.")
            return []
        print(f"    batch of {len(batch)} failed ({str(err)[:80]}); splitting in two.")
        time.sleep(2 * (depth + 1))
        half = len(batch) // 2
        return (process_terrain_batch(ee, images, batch.iloc[:half], tile_scale, depth + 1)
                + process_terrain_batch(ee, images, batch.iloc[half:], tile_scale, depth + 1))
    return [assemble_terrain_row(pid, means.get(pid, {}), cover.get(pid, {})) for pid in ids]


def finished_ids(path):
    """Parcel ids already in an output CSV (for resuming)."""
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["parcel_id"] for row in csv.DictReader(f)}


def reduce_shapes(ee, image, ids, shapes, tile_scale, reducer=None):
    """Summarize the image over each shape (mean unless another reducer is
    given). Returns {parcel_id: properties}."""
    features = [ee.Feature(to_ee_geometry(ee, s), {"parcel_id": pid})
                for pid, s in zip(ids, shapes)]
    # reduceRegions = "for every shape, summarize the pixels inside it".
    # ee.Reducer.mean() weights edge pixels by the fraction inside the shape.
    # A mosaic has no native pixel grid, so scale and crs must be given.
    # tileScale > 1 trades speed for memory and avoids out-of-memory errors.
    reduced = image.reduceRegions(
        collection=ee.FeatureCollection(features),
        reducer=ee.Reducer.mean() if reducer is None else reducer,
        scale=PIXEL_M, crs=WORK_CRS, tileScale=tile_scale)
    # Drop the geometry from the reply; we only need the numbers back.
    info = reduced.select(propertySelectors=[".*"], retainGeometry=False).getInfo()
    return {f["properties"]["parcel_id"]: f["properties"] for f in info["features"]}


def process_batch(ee, image, batch, tile_scale, depth=0):
    """Run one batch. If Earth Engine rejects it (too large, timed out),
    split it in half and try again, down to single parcels."""
    ids = batch["parcel_id"].tolist()
    try:
        parcel = reduce_shapes(ee, image, ids, batch["sample_ll"], tile_scale)
        ring = reduce_shapes(ee, image, ids, batch["ring_ll"], tile_scale)
    except Exception as err:  # ee.EEException and network errors
        if len(batch) == 1:
            # Not written to the CSV, so a later re-run will try it again.
            print(f"    WARNING: parcel {ids[0]} failed ({str(err)[:120]}); skipped.")
            return []
        else:
            print(f"    batch of {len(batch)} failed ({str(err)[:80]}); splitting in two.")
            time.sleep(2 * (depth + 1))
            half = len(batch) // 2
            return (process_batch(ee, image, batch.iloc[:half], tile_scale, depth + 1)
                    + process_batch(ee, image, batch.iloc[half:], tile_scale, depth + 1))
    return [assemble_row(r.parcel_id, r.is_tiny, r.area_m2,
                         parcel.get(r.parcel_id, {}), ring.get(r.parcel_id, {}))
            for r in batch.itertuples()]


def main():
    ap = argparse.ArgumentParser(description="Extract AlphaEarth parcel features with Earth Engine.")
    ap.add_argument("--project", required=True, help="Google Cloud project ID registered for Earth Engine.")
    ap.add_argument("--parcels", required=True, help="Parcels GeoJSON (or GeoPackage) with an id column.")
    ap.add_argument("--id-field", default="parcel_id", help="Name of the parcel id column.")
    ap.add_argument("--out", default="embeddings.csv")
    ap.add_argument("--pilot", type=int, default=0, help="Only process the first N parcels.")
    ap.add_argument("--year", type=int, default=2020, help="Embedding year for the features.")
    ap.add_argument("--change-from-year", type=int, default=2018, help="Earlier year for the change feature.")
    ap.add_argument("--cutoff-year", type=int, default=2020,
                    help="Latest year allowed. Features after this would leak the label.")
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--tile-scale", type=int, default=4)
    ap.add_argument("--terrain-out", default=None,
                    help="Also write elevation, slope and NLCD land cover per parcel to this CSV.")
    ap.add_argument("--terrain-only", action="store_true",
                    help="Skip the embeddings and only write --terrain-out.")
    args = ap.parse_args()

    if args.terrain_only and not args.terrain_out:
        sys.exit("ERROR: --terrain-only needs --terrain-out.")
    if args.terrain_out and NLCD_YEAR > args.cutoff_year:
        sys.exit(f"ERROR: NLCD_YEAR ({NLCD_YEAR}) is after --cutoff-year ({args.cutoff_year}).")

    if args.year > args.cutoff_year or args.change_from_year >= args.year:
        sys.exit(f"ERROR: --year ({args.year}) must be <= --cutoff-year ({args.cutoff_year}) and "
                 f"--change-from-year ({args.change_from_year}) must be earlier than --year. "
                 "Imagery from after the cutoff can show the outcome you are predicting.")
    if args.change_from_year < 2017:
        sys.exit("ERROR: the Satellite Embedding dataset starts in 2017.")

    gdf, bounds = prepare_parcels(args.parcels, args.id_field, args.pilot)
    print(f"{len(gdf):,} parcels to process; {int(gdf['is_tiny'].sum()):,} are tiny "
          f"(< {TINY_AREA_M2} m^2) and use the centroid fallback.")

    # Resume support: skip parcels that are already in the output file.
    out = Path(args.out)
    done = set() if args.terrain_only else finished_ids(out)
    if done:
        print(f"Found {len(done):,} finished parcels in {out}; skipping those.")
    todo = gdf[~gdf["parcel_id"].isin(done)]

    try:
        import ee
    except ImportError:
        sys.exit("This script needs the Earth Engine API:  pip install earthengine-api")
    try:
        ee.Initialize(project=args.project)
    except Exception as err:
        sys.exit(f"Could not start Earth Engine: {err}\n"
                 "Run `earthengine authenticate` once, and check that the project "
                 "ID is registered for Earth Engine.")

    if not args.terrain_only:
        image = build_image(ee, args.year, args.change_from_year, bounds)

        write_header = not out.exists() or out.stat().st_size == 0
        with open(out, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
            if write_header:
                writer.writeheader()
            n_batches = int(np.ceil(len(todo) / args.batch_size))
            for i in range(n_batches):
                batch = todo.iloc[i * args.batch_size:(i + 1) * args.batch_size]
                t0 = time.time()
                rows = process_batch(ee, image, batch, args.tile_scale)
                writer.writerows(rows)
                f.flush()   # so a crash never loses finished batches
                blanks = sum(np.isnan(r["emb_A00"]) for r in rows)
                skipped = len(batch) - len(rows)
                print(f"  batch {i + 1}/{n_batches}: {len(rows)} parcels in {time.time() - t0:.0f}s"
                      + (f"  ({blanks} with no data)" if blanks else "")
                      + (f"  ({skipped} failed; re-run to retry)" if skipped else ""))

        print(f"\nDone. Wrote {out}")
        print("Check before the full run: open the CSV, confirm emb_A00..A63 are filled, "
              "values sit roughly between -1 and 1, and emb_change is a small number >= 0.")

    if args.terrain_out:
        # Same pattern as above, for elevation, slope and land cover.
        t_out = Path(args.terrain_out)
        t_done = finished_ids(t_out)
        if t_done:
            print(f"Found {len(t_done):,} finished parcels in {t_out}; skipping those.")
        t_todo = gdf[~gdf["parcel_id"].isin(t_done)]
        images = build_terrain_images(ee)
        write_header = not t_out.exists() or t_out.stat().st_size == 0
        with open(t_out, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TERRAIN_COLUMNS)
            if write_header:
                writer.writeheader()
            n_batches = int(np.ceil(len(t_todo) / args.batch_size))
            for i in range(n_batches):
                batch = t_todo.iloc[i * args.batch_size:(i + 1) * args.batch_size]
                t0 = time.time()
                rows = process_terrain_batch(ee, images, batch, args.tile_scale)
                writer.writerows(rows)
                f.flush()
                blanks = sum(np.isnan(r["elevation"]) for r in rows)
                skipped = len(batch) - len(rows)
                print(f"  terrain batch {i + 1}/{n_batches}: {len(rows)} parcels in {time.time() - t0:.0f}s"
                      + (f"  ({blanks} with no data)" if blanks else "")
                      + (f"  ({skipped} failed; re-run to retry)" if skipped else ""))
        print(f"\nDone. Wrote {t_out}")
        print("Check before the full run: elevation near Austin should sit roughly between "
              "120 and 400 m, slope is a percent >= 0, and land_cover_class is a name, "
              "not 'missing' or 'nlcd_<number>'.")

    print('Attribution: "The AlphaEarth Foundations Satellite Embedding dataset is '
          'produced by Google and Google DeepMind."')


if __name__ == "__main__":
    main()
