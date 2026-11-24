"""
Tests for build_features.py on small FAKE geometries (squares on a grid).

AI NOTE: written with Claude (Anthropic). Nothing here is real Austin data.
Run from site_suitability_starter/:   python -m pytest tests -q
"""
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import build_features as bf  # noqa: E402

UTM = "EPSG:32614"
X0, Y0 = 620_000.0, 3_350_000.0      # roughly Austin, in meters


def square(col, row, size=100.0, gap=100.0):
    x = X0 + col * (size + gap)
    y = Y0 + row * (size + gap)
    return box(x, y, x + size, y + size)


@pytest.fixture
def parcels():
    # Four 100 m x 100 m lots in a row, 100 m apart.
    return gpd.GeoDataFrame(
        {"parcel_id": ["A", "B", "C", "D"], "PROP_ID": ["0011", "12", "13", "14"],
         "developed": [1, 0, 0, 1]},
        geometry=[square(i, 0) for i in range(4)], crs=UTM)


# ---- pure helpers ------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("SF-3-NP", "SF-3"), ("CS-MU-V-CO-NP", "CS"), ("MF-4-CO", "MF-4"),
    ("SF-4A", "SF-4A"), ("I-SF-2", "SF-2"), ("PUD", "PUD"), ("gr-mu", "GR"),
    ("LI-PDA", "LI"), (" CBD-CURE ", "CBD"), ("", "missing"), (None, "missing"),
    (float("nan"), "missing"), ("123", "missing"),
])
def test_base_zoning(raw, expected):
    assert bf.base_zoning(raw) == expected


def test_improvement_ratio():
    r = bf.improvement_ratio([100, 0, 50, None, 0], [0, 100, 150, 10, 0])
    assert r[0] == 0.0 and r[1] == 1.0 and r[2] == 0.75
    assert np.isnan(r[3]) and np.isnan(r[4])     # missing value, zero total


def test_norm_id_matches_across_formats():
    s = bf.norm_id(pd.Series(["0011", " 12 ", 13, "14.0", "000"]))
    assert list(s) == ["11", "12", "13", "14", "0"]


def test_check_year_refuses_post_cutoff():
    bf.check_year("x", 2020, 2020)
    bf.check_year("x", None, 2020)
    with pytest.raises(SystemExit):
        bf.check_year("x", 2021, 2020)


# ---- geometry features -------------------------------------------------------
def test_lot_area(parcels):
    assert np.allclose(bf.lot_area(parcels), 10_000.0)


def test_zoning_uses_interior_point_and_collapses(parcels):
    # One zone over A and B, another over C. D is under nothing.
    zones = gpd.GeoDataFrame(
        {"ZONING_ZTYPE": ["SF-3-NP", "CS-MU-CO"]},
        geometry=[box(X0 - 10, Y0 - 10, X0 + 310, Y0 + 110),
                  box(X0 + 390, Y0 - 10, X0 + 510, Y0 + 110)], crs=UTM).to_crs(4326)
    got = bf.zoning_class(parcels, zones, "ZONING_ZTYPE")
    assert list(got) == ["SF-3", "SF-3", "CS", "missing"]


def test_floodplain_share(parcels):
    # Covers the left half of A, all of B, and a 0.5 m sliver of C. Two
    # overlapping pieces over B must not count twice.
    flood = gpd.GeoDataFrame(geometry=[
        box(X0 - 50, Y0 - 50, X0 + 50, Y0 + 150),
        box(X0 + 190, Y0 - 10, X0 + 310, Y0 + 110),
        box(X0 + 190, Y0 - 10, X0 + 310, Y0 + 110),
        box(X0 + 399.5, Y0 - 10, X0 + 400.5, Y0 + 110),
    ], crs=UTM)
    share = bf.floodplain_share(parcels, flood)
    assert share.iloc[0] == pytest.approx(0.5)
    assert share.iloc[1] == pytest.approx(1.0)
    assert share.iloc[2] == pytest.approx(0.005)
    assert share.iloc[3] == 0.0
    assert list((share > 0.01).astype(int)) == [1, 1, 0, 0]   # sliver ignored


def test_dist_to_road(parcels):
    # A horizontal road 30 m south of every lot, and one crossing lot D.
    roads = gpd.GeoDataFrame(geometry=[
        LineString([(X0 - 500, Y0 - 30), (X0 + 1500, Y0 - 30)]),
        LineString([(X0 + 650, Y0 - 100), (X0 + 650, Y0 + 300)]),
    ], crs=UTM).to_crs(4326)
    d = bf.dist_to_road(parcels, roads)
    assert d.iloc[0] == pytest.approx(30, abs=0.05)
    assert d.iloc[1] == pytest.approx(30, abs=0.05)
    assert d.iloc[3] == pytest.approx(0, abs=0.05)


def test_inside_limits(parcels):
    limits = gpd.GeoDataFrame(geometry=[box(X0 - 10, Y0 - 10, X0 + 350, Y0 + 110)], crs=UTM)
    assert list(bf.inside_limits(parcels, limits)) == [True, True, False, False]


def test_appraisal_ratio_sums_duplicate_rows():
    roll = pd.DataFrame({"prop_id": ["11", "11", "12"], "land_value": ["50", "50", "100"],
                         "improvement_value": ["0", "100", "300"]})
    r = bf.appraisal_ratio(roll, "prop_id", "land_value", "improvement_value")
    assert r["11"] == pytest.approx(0.5) and r["12"] == pytest.approx(0.75)


# ---- the whole script, end to end --------------------------------------------
def run(args, cwd):
    return subprocess.run([sys.executable, str(HERE / "build_features.py")] + args,
                          cwd=cwd, capture_output=True, text=True)


@pytest.fixture
def files(tmp_path, parcels):
    parcels.to_crs(4326).to_file(tmp_path / "labels.geojson", driver="GeoJSON")
    pd.DataFrame({"prop_id": ["11", "12", "13", "14"],
                  "land_value": [100, 100, 100, 100],
                  "improvement_value": [0, 10, 900, 20]}).to_csv(tmp_path / "roll.csv", index=False)
    emb = pd.DataFrame({"parcel_id": ["A", "B", "D"], "is_tiny_parcel": [0, 0, 0]})
    for i in range(64):
        emb[f"emb_A{i:02d}"] = 0.1
    emb.to_csv(tmp_path / "emb.csv", index=False)
    pd.DataFrame({"parcel_id": ["A", "B", "C", "D"], "elevation": [200.0, 201, 202, 203],
                  "slope": [1.0, 2, 3, 4], "land_cover_class": ["grassland"] * 4}
                 ).to_csv(tmp_path / "terrain.csv", index=False)
    return tmp_path


def test_end_to_end_filters_and_joins(files):
    r = run(["--labels", "labels.geojson", "--parcel-year", "2020",
             "--appraisal", "roll.csv", "--appraisal-year", "2020",
             "--labels-join-col", "PROP_ID",
             "--terrain", "terrain.csv", "--embeddings", "emb.csv",
             "--out", "features.csv"], files)
    assert r.returncode == 0, r.stderr
    f = pd.read_csv(files / "features.csv", dtype={"parcel_id": str})
    # C has ratio 0.9 (built up) and is filtered out. "0011" matched "11".
    assert list(f["parcel_id"]) == ["A", "B", "D"]
    assert f["improvement_value_ratio"].round(4).tolist() == [0.0, round(10 / 110, 4), round(20 / 120, 4)]
    assert np.allclose(f["lot_area"], 10_000, rtol=1e-3)
    assert f["x"].between(600_000, 640_000).all() and f["y"].between(3_300_000, 3_400_000).all()
    assert {"slope", "elevation", "land_cover_class", "emb_A00", "emb_A63"} <= set(f.columns)
    assert "zoning_class" not in f.columns          # layer not given -> column left out
    assert "--numeric-cols" in r.stdout             # and the script says how to tell train_models


def test_refuses_post_cutoff_appraisal(files):
    r = run(["--labels", "labels.geojson", "--parcel-year", "2020",
             "--appraisal", "roll.csv", "--appraisal-year", "2023", "--out", "f.csv"], files)
    assert r.returncode != 0 and "after the cutoff" in r.stderr
    assert not (files / "f.csv").exists()


def test_requires_parcel_year(files):
    r = run(["--labels", "labels.geojson", "--out", "f.csv"], files)
    assert r.returncode != 0 and "--parcel-year" in r.stderr
