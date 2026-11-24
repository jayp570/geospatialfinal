"""
make_synthetic_data.py -- build a FAKE parcel dataset for smoke-testing.

==========================================================================
  EVERYTHING THIS SCRIPT PRODUCES IS SYNTHETIC. It is not Austin data and
  no result computed from it says anything about real development.
  The output carries an `is_synthetic` column so train_models.py stamps
  "SYNTHETIC" on every table and figure it writes.
==========================================================================

Why have fake data at all?
    The real pipeline needs downloads and an Earth Engine account. Fake data
    with a KNOWN answer lets us check that train_models.py behaves correctly
    before any real data exists:
      * if we plant an imagery signal, the imagery model should find it;
      * if we plant none (--imagery-signal 0), the imagery model should NOT
        appear to help. If it does, the evaluation is fooling us.

How the fake world works (so you can reason about the test results):

  1. Parcels sit in "neighborhoods" (small clusters, a few hundred meters
     across). Real development is clustered the same way: a builder buys a
     tract and develops many adjacent lots together.

  2. Whether a parcel gets developed depends on
        a. tabular facts we observe (lot size, zoning, floodplain, slope ...),
        b. a smooth city-wide "growth pressure" surface,
        c. a hidden NEIGHBORHOOD effect (momentum: is this area taking off?),
        d. a hidden PARCEL effect (site readiness: cleared, graded, accessible).
     (c) and (d) are not in the tabular columns. They are the things a
     satellite image could plausibly reveal.

  3. Each parcel gets a 64-number "embedding" shaped like the real AlphaEarth
     ones (unit-length vectors). Every embedding contains
        - a neighborhood "fingerprint": random numbers shared by all parcels
          in the same neighborhood. This mimics the fact that neighboring
          parcels look alike from above. It carries NO information about
          development by itself.
        - a land-cover component (imagery can see land cover; this is
          redundant with the tabular land_cover_class column).
        - the hidden effects (c) and (d), scaled by --imagery-signal.

     With --imagery-signal 0 the embeddings are pure fingerprint + land cover.
     A model can still "memorize" which fingerprints belong to booming
     neighborhoods, and that memory helps it on other parcels from the SAME
     neighborhood. This is the spatial-autocorrelation trap: random
     cross-validation rewards it, spatial block cross-validation does not.

Usage:
    python make_synthetic_data.py --out synthetic_demo/features_signal_on.csv
    python make_synthetic_data.py --imagery-signal 0 \
        --out synthetic_demo/features_signal_off.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EMBED_DIM = 64  # AlphaEarth embeddings have 64 bands, A00..A63

# Rough Austin location in UTM zone 14N (EPSG:32614), in meters. Only used so
# the fake coordinates look like the real ones will.
ORIGIN_X, ORIGIN_Y = 608_000.0, 3_336_000.0
EXTENT_X, EXTENT_Y = 24_000.0, 28_000.0  # a 24 km x 28 km study area

# Zoning districts with a made-up effect on development odds.
ZONING_EFFECT = {
    "SF-2": 0.00, "SF-3": 0.25, "MF-3": 0.60, "GR": 0.35,
    "CS": 0.50, "LI": -0.10, "P": -1.50,
}
ZONING_PROBS = [0.26, 0.28, 0.12, 0.10, 0.12, 0.07, 0.05]

LAND_COVER_EFFECT = {
    "developed_open": 0.30, "grassland": 0.20, "shrub": 0.00,
    "forest": -0.30, "cultivated": 0.10,
}
LAND_COVER_PROBS = [0.22, 0.30, 0.18, 0.22, 0.08]


def smooth_field(x, y, rng, n_bumps, length_scale):
    """A smooth random surface: the sum of a few Gaussian hills and valleys.

    Nearby points get similar values, which is what "spatially
    autocorrelated" means. Returned standardized (mean 0, std 1).
    """
    cx = rng.uniform(0, EXTENT_X, n_bumps)
    cy = rng.uniform(0, EXTENT_Y, n_bumps)
    amp = rng.normal(0, 1, n_bumps)
    d2 = (x[:, None] - cx[None, :]) ** 2 + (y[:, None] - cy[None, :]) ** 2
    f = (amp[None, :] * np.exp(-d2 / (2 * length_scale ** 2))).sum(axis=1)
    return (f - f.mean()) / f.std()


def zscore(v):
    return (v - np.nanmean(v)) / np.nanstd(v)


def unit_rows(m):
    """Scale every row to length 1, like the real AlphaEarth vectors."""
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def random_direction(rng):
    d = rng.normal(0, 1, EMBED_DIM)
    return d / np.linalg.norm(d)


def calibrate_intercept(logit_no_intercept, target_rate):
    """Find the intercept that makes the average probability hit the target."""
    lo, hi = -15.0, 15.0
    for _ in range(60):  # bisection
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(logit_no_intercept + mid)))).mean()
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    return (lo + hi) / 2


def make_dataset(n_parcels, n_neighborhoods, imagery_signal, positive_rate,
                 clay_dims, seed):
    rng = np.random.default_rng(seed)

    # ---- 1. Where things are ------------------------------------------------
    hood_x = rng.uniform(0, EXTENT_X, n_neighborhoods)
    hood_y = rng.uniform(0, EXTENT_Y, n_neighborhoods)
    hood = rng.integers(0, n_neighborhoods, n_parcels)      # parcel -> hood
    x = hood_x[hood] + rng.normal(0, 250, n_parcels)        # ~250 m spread
    y = hood_y[hood] + rng.normal(0, 250, n_parcels)

    # ---- 2. Hidden drivers the tabular columns do not contain ---------------
    hood_momentum = rng.normal(0, 1, n_neighborhoods)       # (c) neighborhood
    site_ready = rng.normal(0, 1, n_parcels)                # (d) parcel
    pressure = smooth_field(x, y, rng, n_bumps=14, length_scale=5_000)  # (b)

    # ---- 3. Tabular features (what the parcel-only model sees) --------------
    lot_area = np.exp(rng.normal(np.log(900), 0.9, n_parcels))       # m^2
    hood_zone = rng.choice(len(ZONING_EFFECT), n_neighborhoods, p=ZONING_PROBS)
    zone_idx = np.where(rng.random(n_parcels) < 0.75, hood_zone[hood],
                        rng.choice(len(ZONING_EFFECT), n_parcels, p=ZONING_PROBS))
    zoning = np.array(list(ZONING_EFFECT))[zone_idx]

    flood_surface = smooth_field(x, y, rng, n_bumps=30, length_scale=1_500)
    in_floodplain = (flood_surface > np.quantile(flood_surface, 0.92)).astype(int)

    slope = rng.gamma(shape=2.0, scale=2.0, size=n_parcels)          # percent
    elevation = 180 + 35 * smooth_field(x, y, rng, 10, 6_000) + rng.normal(0, 4, n_parcels)
    dist_to_road = rng.exponential(80, n_parcels)                    # meters
    improvement_value_ratio = rng.beta(1.2, 6.0, n_parcels)          # low = underused
    lc_idx = rng.choice(len(LAND_COVER_EFFECT), n_parcels, p=LAND_COVER_PROBS)
    land_cover = np.array(list(LAND_COVER_EFFECT))[lc_idx]

    # ---- 4. The label --------------------------------------------------------
    logit = (
        0.50 * zscore(np.log(lot_area))
        + np.array([ZONING_EFFECT[z] for z in zoning])
        - 1.20 * in_floodplain
        - 0.35 * zscore(slope)
        - 0.40 * zscore(np.log1p(dist_to_road))
        - 0.60 * zscore(improvement_value_ratio)
        + np.array([LAND_COVER_EFFECT[c] for c in land_cover])
        + 0.60 * pressure
        + 0.90 * hood_momentum[hood]      # hidden
        + 0.70 * site_ready               # hidden
    )
    logit += calibrate_intercept(logit, positive_rate)
    developed = (rng.random(n_parcels) < 1 / (1 + np.exp(-logit))).astype(int)

    # ---- 5. Fake AlphaEarth-style embeddings --------------------------------
    scale = 1 / np.sqrt(EMBED_DIM)
    fp_parcel = rng.normal(0, scale, (n_neighborhoods, EMBED_DIM))   # fingerprints
    fp_nbr = rng.normal(0, scale, (n_neighborhoods, EMBED_DIM))
    lc_dirs = rng.normal(0, scale, (len(LAND_COVER_EFFECT), EMBED_DIM))
    d_site, d_mom, d_nbr = (random_direction(rng) for _ in range(3))
    mom = hood_momentum[hood][:, None]

    emb = unit_rows(
        0.8 * fp_parcel[hood]
        + 0.6 * rng.normal(0, scale, (n_parcels, EMBED_DIM))
        + 0.5 * lc_dirs[lc_idx]
        + imagery_signal * (0.45 * site_ready[:, None] * d_site + 0.35 * mom * d_mom)
    )
    nbr = unit_rows(
        1.0 * fp_nbr[hood]
        + 0.15 * rng.normal(0, scale, (n_parcels, EMBED_DIM))
        + imagery_signal * 0.5 * mom * d_nbr
    )
    # "How much did this place change between 2018 and 2020?" In the real
    # pipeline this is 1 - cosine similarity, a small non-negative number.
    relu = lambda v: np.maximum(v, 0)
    emb_change = np.clip(
        0.02 + 0.015 * np.abs(rng.normal(0, 1, n_parcels))
        + imagery_signal * 0.02 * relu(0.6 * site_ready + 0.4 * hood_momentum[hood]),
        0, 1)
    nbr_change = np.clip(
        0.015 + 0.008 * np.abs(rng.normal(0, 1, n_parcels))
        + imagery_signal * 0.015 * relu(hood_momentum[hood]),
        0, 1)

    # ---- 6. Assemble ---------------------------------------------------------
    df = pd.DataFrame({
        "parcel_id": [f"SYN{i:06d}" for i in range(n_parcels)],
        "x": np.round(ORIGIN_X + x, 1),
        "y": np.round(ORIGIN_Y + y, 1),
        "developed": developed,
        "lot_area": np.round(lot_area, 1),
        "zoning_class": zoning,
        "in_floodplain": in_floodplain,
        "slope": np.round(slope, 2),
        "elevation": np.round(elevation, 1),
        "dist_to_road": np.round(dist_to_road, 1),
        "land_cover_class": land_cover,
        "improvement_value_ratio": np.round(improvement_value_ratio, 4),
        # Parcels smaller than about 3 AlphaEarth pixels (10 m x 10 m each).
        "is_tiny_parcel": (lot_area < 300).astype(int),
    })
    # Real data has gaps; add a few so the imputation code is exercised.
    for col in ("slope", "improvement_value_ratio"):
        df.loc[rng.random(n_parcels) < 0.02, col] = np.nan

    blocks = [df]
    blocks.append(pd.DataFrame(np.round(emb, 5),
                               columns=[f"emb_A{i:02d}" for i in range(EMBED_DIM)]))
    blocks.append(pd.DataFrame(np.round(nbr, 5),
                               columns=[f"nbr_A{i:02d}" for i in range(EMBED_DIM)]))
    blocks.append(pd.DataFrame({"emb_change": np.round(emb_change, 5),
                                "nbr_change": np.round(nbr_change, 5)}))

    if clay_dims > 0:
        # Optional stand-in for Clay chip embeddings. A ~150 m chip mostly
        # shows the surroundings, so the parcel-level signal is weak.
        c_scale = 1 / np.sqrt(clay_dims)
        fp_clay = rng.normal(0, c_scale, (n_neighborhoods, clay_dims))
        dc_site, dc_mom = (rng.normal(0, c_scale, clay_dims) for _ in range(2))
        clay = (
            1.0 * fp_clay[hood]
            + 0.3 * rng.normal(0, c_scale, (n_parcels, clay_dims))
            + imagery_signal * (0.15 * site_ready[:, None] * dc_site + 0.35 * mom * dc_mom)
        )
        blocks.append(pd.DataFrame(np.round(clay, 5),
                                   columns=[f"clay_{i:03d}" for i in range(clay_dims)]))

    out = pd.concat(blocks, axis=1)
    out["is_synthetic"] = 1  # train_models.py looks for this
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="synthetic_demo/features_synthetic.csv")
    ap.add_argument("--n-parcels", type=int, default=9000)
    ap.add_argument("--n-neighborhoods", type=int, default=320)
    ap.add_argument("--imagery-signal", type=float, default=1.0,
                    help="How strongly the fake embeddings encode the hidden "
                         "drivers of development. 0 = no real signal.")
    ap.add_argument("--positive-rate", type=float, default=0.12,
                    help="Target share of parcels with developed = 1.")
    ap.add_argument("--clay-dims", type=int, default=0,
                    help="If > 0, also write this many fake clay_### columns.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = make_dataset(args.n_parcels, args.n_neighborhoods, args.imagery_signal,
                      args.positive_rate, args.clay_dims, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"SYNTHETIC data written to {out}")
    print(f"  parcels: {len(df):,}   columns: {df.shape[1]}   "
          f"positive rate: {df['developed'].mean():.3f}   "
          f"imagery signal: {args.imagery_signal}")


if __name__ == "__main__":
    main()
