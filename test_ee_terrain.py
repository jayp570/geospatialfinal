"""
Tests for the terrain / land-cover additions to ee_extract_embeddings.py.

AI NOTE: written with Claude (Anthropic). These tests do NOT contact Earth
Engine. They use a stand-in ("mock") for the `ee` library, so they prove the
script's own logic and file handling, not that Earth Engine returns what we
expect. Only the --pilot run can prove that.
Run from site_suitability_starter/:   python -m pytest tests -q
"""
import sys
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import ee_extract_embeddings as eex  # noqa: E402


@pytest.mark.parametrize("code,name", [
    (71, "grassland"), (71.0, "grassland"), ("42", "evergreen_forest"),
    (21, "developed_open_space"), (None, "missing"), (float("nan"), "missing"),
    ("abc", "missing"), (99, "nlcd_99"),
])
def test_nlcd_class_name(code, name):
    assert eex.nlcd_class_name(code) == name


def test_nlcd_year_is_before_cutoff():
    assert eex.NLCD_YEAR <= 2020


def test_assemble_terrain_row_accepts_both_property_names():
    row = eex.assemble_terrain_row("P1", {"elevation": 201.456, "slope": 3.14159}, {"mode": 71})
    assert row == {"parcel_id": "P1", "elevation": 201.46, "slope": 3.142,
                   "nlcd_code": 71, "land_cover_class": "grassland"}
    assert eex.assemble_terrain_row("P1", {}, {"landcover": 23})["land_cover_class"] == "developed_medium"
    assert list(row) == eex.TERRAIN_COLUMNS


def test_assemble_terrain_row_missing_values():
    row = eex.assemble_terrain_row("P2", {}, {})
    assert np.isnan(row["elevation"]) and np.isnan(row["slope"])
    assert row["nlcd_code"] == "" and row["land_cover_class"] == "missing"


def fake_parcels(tmp_path, n=5):
    # n fake 60 m squares near Austin, written as longitude/latitude GeoJSON.
    geoms = [box(620_000 + 200 * i, 3_350_000, 620_060 + 200 * i, 3_350_060) for i in range(n)]
    gdf = gpd.GeoDataFrame({"parcel_id": [f"P{i}" for i in range(n)]}, geometry=geoms, crs="EPSG:32614")
    path = tmp_path / "parcels.geojson"
    gdf.to_crs(4326).to_file(path, driver="GeoJSON")
    return path


def fake_reduce(calls):
    def reduce_shapes(ee, image, ids, shapes, tile_scale, reducer=None):
        calls.append((len(ids), reducer is not None))
        if reducer is not None:                                   # land cover
            return {pid: {"mode": 71} for pid in ids}
        props = {"elevation": 200.0, "slope": 2.5}                # terrain means
        props.update({f"now_{b}": 0.125 for b in eex.BANDS})      # embeddings
        props.update({f"then_{b}": 0.125 for b in eex.BANDS})
        return {pid: dict(props) for pid in ids}
    return reduce_shapes


def run_main(monkeypatch, argv, calls):
    monkeypatch.setitem(sys.modules, "ee", mock.MagicMock())
    monkeypatch.setattr(eex, "reduce_shapes", fake_reduce(calls))
    monkeypatch.setattr(sys, "argv", ["ee_extract_embeddings.py"] + argv)
    eex.main()


def test_main_writes_embeddings_and_terrain(tmp_path, monkeypatch):
    parcels, calls = fake_parcels(tmp_path), []
    out, t_out = tmp_path / "emb.csv", tmp_path / "terrain.csv"
    run_main(monkeypatch, ["--project", "x", "--parcels", str(parcels), "--out", str(out),
                           "--terrain-out", str(t_out), "--batch-size", "2"], calls)
    emb, ter = pd.read_csv(out), pd.read_csv(t_out)
    assert list(emb.columns) == eex.OUT_COLUMNS and len(emb) == 5
    assert list(ter.columns) == eex.TERRAIN_COLUMNS and len(ter) == 5
    assert (ter["land_cover_class"] == "grassland").all() and (ter["slope"] == 2.5).all()
    assert emb["emb_change"].abs().max() < 1e-9          # identical years -> no change

    # Resume: a second run finds everything finished and asks Earth Engine nothing.
    calls.clear()
    run_main(monkeypatch, ["--project", "x", "--parcels", str(parcels), "--out", str(out),
                           "--terrain-out", str(t_out)], calls)
    assert calls == [] and len(pd.read_csv(t_out)) == 5


def test_terrain_only_skips_embeddings(tmp_path, monkeypatch):
    parcels, calls = fake_parcels(tmp_path), []
    out, t_out = tmp_path / "emb.csv", tmp_path / "terrain.csv"
    run_main(monkeypatch, ["--project", "x", "--parcels", str(parcels), "--out", str(out),
                           "--terrain-out", str(t_out), "--terrain-only"], calls)
    assert not out.exists() and len(pd.read_csv(t_out)) == 5


def test_terrain_only_needs_terrain_out(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, ["--project", "x", "--parcels", "none.geojson", "--terrain-only"], [])


def test_terrain_batch_splits_on_failure(monkeypatch):
    monkeypatch.setattr(eex.time, "sleep", lambda s: None)
    good = fake_reduce([])

    def flaky(ee, image, ids, shapes, tile_scale, reducer=None):
        if len(ids) > 1:
            raise RuntimeError("payload too large")
        if ids == ["P2"]:
            raise RuntimeError("bad geometry")
        return good(ee, image, ids, shapes, tile_scale, reducer)

    monkeypatch.setattr(eex, "reduce_shapes", flaky)
    batch = pd.DataFrame({"parcel_id": ["P0", "P1", "P2", "P3"], "sample_ll": [None] * 4})
    rows = eex.process_terrain_batch(mock.MagicMock(), (None, None), batch, 4)
    assert [r["parcel_id"] for r in rows] == ["P0", "P1", "P3"]     # P2 skipped, rest kept
