"""
Tests for the helper functions in ee_extract_embeddings.py.

AI NOTE: written with Claude (Anthropic). These tests do NOT contact Earth
Engine. Where the script talks to Earth Engine, a stand-in ("mock") is used,
so they check the script's own arithmetic, geometry and retry logic only.
Run from site_suitability_starter/:   python -m pytest tests -q
"""
import subprocess
import sys
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, shape

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import ee_extract_embeddings as eex  # noqa: E402

UTM = "EPSG:32614"
X0, Y0 = 620_000.0, 3_350_000.0


def props(prefix, value):
    return {f"{prefix}_{b}": value for b in eex.BANDS}


# ---- arithmetic --------------------------------------------------------------
def test_vector_reads_64_bands_and_marks_missing():
    p = props("now", 0.5)
    p["now_A07"] = None
    del p["now_A09"]
    v = eex.vector(p, "now")
    assert v.shape == (64,) and v[0] == 0.5
    assert np.isnan(v[7]) and np.isnan(v[9]) and np.isnan(v).sum() == 2


def test_cosine_change_known_values():
    a = np.zeros(64); a[0] = 1.0
    b = np.zeros(64); b[1] = 1.0
    assert eex.cosine_change(a, a) == pytest.approx(0.0)        # unchanged
    assert eex.cosine_change(a, 3 * a) == pytest.approx(0.0)    # length does not matter
    assert eex.cosine_change(a, b) == pytest.approx(1.0)        # unrelated
    assert eex.cosine_change(a, -a) == pytest.approx(2.0)       # opposite


def test_cosine_change_missing_or_zero_is_nan():
    a = np.ones(64)
    bad = a.copy(); bad[3] = np.nan
    assert np.isnan(eex.cosine_change(bad, a)) and np.isnan(eex.cosine_change(a, bad))
    assert np.isnan(eex.cosine_change(np.zeros(64), a))


def test_assemble_row_has_exactly_the_output_columns():
    now = props("now", 0.125); now.update(props("then", 0.125))
    row = eex.assemble_row("P1", True, 123.456, now, now)
    assert list(row) == eex.OUT_COLUMNS
    assert row["is_tiny_parcel"] == 1 and row["parcel_area_m2"] == 123.5
    assert row["emb_A00"] == 0.125 and row["nbr_A63"] == 0.125
    assert row["emb_change"] == pytest.approx(0.0) and row["nbr_change"] == pytest.approx(0.0)


def test_assemble_row_with_no_data_is_all_nan():
    row = eex.assemble_row("P2", False, 500, {}, {})
    assert np.isnan(row["emb_A00"]) and np.isnan(row["nbr_A00"])
    assert np.isnan(row["emb_change"]) and np.isnan(row["nbr_change"])


def test_output_columns_match_what_train_models_looks_for():
    assert sum(c.startswith("emb_A") for c in eex.OUT_COLUMNS) == 64
    assert sum(c.startswith("nbr_A") for c in eex.OUT_COLUMNS) == 64
    assert {"emb_change", "nbr_change", "is_tiny_parcel", "parcel_id"} <= set(eex.OUT_COLUMNS)


# ---- geometry preparation ----------------------------------------------------
@pytest.fixture
def parcel_file(tmp_path):
    geoms = [box(X0, Y0, X0 + 100, Y0 + 100),            # big: 10,000 m^2
             box(X0 + 500, Y0, X0 + 510, Y0 + 10),       # tiny: 100 m^2
             box(X0 + 900, Y0, X0 + 1000, Y0 + 100),     # duplicate id of the first
             None]                                       # no shape
    gdf = gpd.GeoDataFrame({"pid": [1, 2, 1, 4]}, geometry=geoms, crs=UTM).to_crs(4326)
    path = tmp_path / "parcels.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


def test_prepare_parcels_cleans_and_flags(parcel_file):
    gdf, bounds = eex.prepare_parcels(parcel_file, "pid", pilot=0)
    assert list(gdf["parcel_id"]) == ["1", "2"]          # text ids; duplicate and empty dropped
    assert list(gdf["is_tiny"]) == [False, True]
    assert gdf["area_m2"].iloc[0] == pytest.approx(10_000, rel=1e-3)
    assert len(bounds) == 4 and -98.5 < bounds[0] < bounds[2] < -96.5 and 29.5 < bounds[1] < bounds[3] < 31


def test_sample_and_ring_shapes(parcel_file):
    gdf, _ = eex.prepare_parcels(parcel_file, "pid", pilot=0)
    to_m = lambda col: gpd.GeoSeries(gdf[col], crs=4326).to_crs(UTM)
    sample, ring, parcel = to_m("sample_ll"), to_m("ring_ll"), gdf.geometry
    # Big parcel: sampled over its own shape. Tiny parcel: a 15 m circle instead.
    assert sample.iloc[0].area == pytest.approx(10_000, rel=1e-3)
    assert sample.iloc[1].area == pytest.approx(np.pi * 15 ** 2, rel=0.05)
    # Ring: reaches 250 m out and has the parcel itself cut out.
    assert ring.iloc[0].intersection(parcel.iloc[0]).area < 1.0
    minx, miny, maxx, maxy = ring.iloc[0].bounds
    assert minx == pytest.approx(X0 - 250, abs=1) and maxx == pytest.approx(X0 + 350, abs=1)
    expected = 600 * 600 - 10_000 - (4 - np.pi) * 250 ** 2      # square with rounded corners, minus the lot
    assert ring.iloc[0].area == pytest.approx(expected, rel=0.03)


def test_pilot_and_bad_id_field(parcel_file):
    gdf, _ = eex.prepare_parcels(parcel_file, "pid", pilot=1)
    assert len(gdf) == 1
    with pytest.raises(SystemExit):
        eex.prepare_parcels(parcel_file, "nope", pilot=0)


def test_to_ee_geometry_passes_plain_geojson():
    fake_ee = mock.MagicMock()
    eex.to_ee_geometry(fake_ee, box(0, 0, 1, 1))
    (arg,), _ = fake_ee.Geometry.call_args
    assert arg["type"] == "Polygon" and isinstance(arg["coordinates"][0][0], list)   # lists, not tuples
    assert shape(arg).equals(box(0, 0, 1, 1))


# ---- retry logic -------------------------------------------------------------
def test_process_batch_splits_until_it_isolates_the_bad_parcel(monkeypatch):
    monkeypatch.setattr(eex.time, "sleep", lambda s: None)
    sizes = []

    def flaky(ee, image, ids, shapes, tile_scale):
        sizes.append(len(ids))
        if len(ids) > 2 or "P3" in ids and len(ids) > 1 or ids == ["P3"]:
            raise RuntimeError("too large")
        out = props("now", 0.1); out.update(props("then", 0.1))
        return {pid: dict(out) for pid in ids}

    monkeypatch.setattr(eex, "reduce_shapes", flaky)
    batch = pd.DataFrame({"parcel_id": [f"P{i}" for i in range(8)], "is_tiny": [False] * 8,
                          "area_m2": [1000.0] * 8, "sample_ll": [None] * 8, "ring_ll": [None] * 8})
    rows = eex.process_batch(mock.MagicMock(), None, batch, 4)
    assert [r["parcel_id"] for r in rows] == ["P0", "P1", "P2", "P4", "P5", "P6", "P7"]
    assert sizes[0] == 8 and 1 in sizes


# ---- the leakage guard on the command line -----------------------------------
@pytest.mark.parametrize("extra,ok_text", [
    (["--year", "2021"], "after the cutoff"),                             # after the cutoff
    (["--year", "2020", "--change-from-year", "2020"], "earlier"),  # change needs an earlier year
    (["--year", "2019", "--change-from-year", "2016"], "2017"),     # dataset starts in 2017
])
def test_refuses_bad_years_before_doing_anything(extra, ok_text):
    r = subprocess.run([sys.executable, str(HERE / "ee_extract_embeddings.py"), "--project", "x",
                        "--parcels", "does_not_exist.geojson"] + extra, capture_output=True, text=True)
    assert r.returncode != 0 and "ERROR" in r.stderr and ok_text in r.stderr
