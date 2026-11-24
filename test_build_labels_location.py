"""
Tests for build_labels.py --label-mode location, on FAKE parcels and permits.

AI NOTE: written with Claude (Anthropic). Nothing here is real Austin data.
Three fake 100 m square lots and a handful of fake permits with known answers.
Run from site_suitability_starter/:   python -m pytest tests -q
"""
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

HERE = Path(__file__).resolve().parent.parent
UTM = "EPSG:32614"
X0, Y0 = 620_000.0, 3_350_000.0


def lot(i):
    return box(X0 + 200 * i, Y0, X0 + 200 * i + 100, Y0 + 100)


def lonlat(x, y):
    p = gpd.GeoSeries.from_xy([x], [y], crs=UTM).to_crs(4326).iloc[0]
    return p.x, p.y


def permit(x, y, date, work="New", cls="Residential", tcad="0000000001"):
    lon, lat = (None, None) if x is None else lonlat(x, y)
    return {"tcad_id": tcad, "issue_date": date, "work_class": work,
            "permit_class_mapped": cls, "latitude": lat, "longitude": lon}


@pytest.fixture
def files(tmp_path):
    parcels = gpd.GeoDataFrame({"GEO": ["0100", "0200", "0300"]},
                               geometry=[lot(0), lot(1), lot(2)], crs=UTM).to_crs(4326)
    parcels.to_file(tmp_path / "parcels.geojson", driver="GeoJSON")
    inside = lambda i: (X0 + 200 * i + 50, Y0 + 50)          # middle of lot i
    rows = [
        permit(*inside(0), "2022-03-01T00:00:00.000"),                    # counts: lot A
        permit(*inside(0), "2024-07-15T00:00:00.000", cls="Commercial"),  # counts: lot A again
        permit(*inside(1), "2020-06-01T00:00:00.000"),                    # before the window
        permit(*inside(1), "2023-01-10T00:00:00.000", work="Remodel"),    # not new construction
        permit(*inside(1), "2023-01-10T00:00:00.000", cls="Sign Permit"), # wrong permit class
        permit(*inside(1), "2026-02-01T00:00:00.000"),                    # after the window
        permit(X0 + 150, Y0 + 50, "2023-05-05T00:00:00.000"),             # in the gap: no lot
        permit(None, None, "2023-05-05T00:00:00.000"),                    # no coordinates
        permit(*inside(2), "2021-01-01T00:00:00.000"),                    # first day of window: lot C
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "permits.csv", index=False)
    return tmp_path


def run(args, cwd):
    return subprocess.run([sys.executable, str(HERE / "build_labels.py")] + args,
                          cwd=cwd, capture_output=True, text=True)


BASE = ["--permits", "permits.csv", "--parcels", "parcels.geojson", "--parcel-id-field", "GEO",
        "--label-mode", "location", "--cutoff-year", "2020", "--negative-sample-ratio", "0",
        "--out", "labeled.geojson"]


def test_location_mode_labels(files):
    r = run(BASE, files)
    assert r.returncode == 0, r.stderr
    out = gpd.read_file(files / "labeled.geojson").set_index("parcel_id")
    assert len(out) == 3                                    # ratio 0 keeps every parcel
    assert out.loc["0100", "developed"] == 1 and out.loc["0100", "permit_count"] == 2
    assert out.loc["0200", "developed"] == 0                # none of its four permits qualify
    assert out.loc["0300", "developed"] == 1                # Jan 1 2021 is inside the window
    assert "5 of 7 qualifying permits fall inside it" in r.stdout
    assert "4 of 5 qualifying permits have coordinates" in r.stdout
    assert "3 permits landed inside a parcel (1 fell outside" in r.stdout


def test_ids_stay_text_and_centroids_are_utm_meters(files):
    run(BASE, files)
    out = gpd.read_file(files / "labeled.geojson")
    assert set(out["parcel_id"]) == {"0100", "0200", "0300"}          # leading zeros kept
    a = out.set_index("parcel_id").loc["0100"]
    assert a["x"] == pytest.approx(X0 + 50, abs=0.5) and a["y"] == pytest.approx(Y0 + 50, abs=0.5)


def test_without_cutoff_every_permit_counts(files):
    args = [a for a in BASE if a not in ("--cutoff-year", "2020")]
    assert run(args, files).returncode == 0
    out = gpd.read_file(files / "labeled.geojson").set_index("parcel_id")
    assert out.loc["0200", "developed"] == 1      # its 2020 and 2026 permits now count


def test_negative_sampling_ratio(files):
    args = BASE.copy()
    args[args.index("--negative-sample-ratio") + 1] = "0.5"
    assert run(args, files).returncode == 0
    out = gpd.read_file(files / "labeled.geojson")
    assert out["developed"].sum() == 2 and (out["developed"] == 0).sum() == 1


def test_wrong_id_field_stops_with_field_list(files):
    args = BASE.copy()
    args[args.index("--parcel-id-field") + 1] = "NOPE"
    r = run(args, files)
    assert r.returncode != 0 and "GEO" in r.stderr


def test_missing_coordinate_column_stops(files):
    r = run(BASE + ["--permit-lat-col", "lat"], files)
    assert r.returncode != 0 and "--permit-lat-col" in r.stderr


def test_warns_when_first_label_year_is_empty(files):
    p = pd.read_csv(files / "permits.csv")
    p[~p["issue_date"].str.startswith("2021")].to_csv(files / "permits.csv", index=False)
    r = run(BASE, files)
    assert r.returncode == 0 and "WARNING: no qualifying permits from 2021" in r.stdout
