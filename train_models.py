"""
train_models.py -- train and fairly compare the three project models.

Research question
    Does adding imagery features from a geospatial foundation model help
    predict which Austin parcels get developed, compared to parcel and
    zoning data alone?

The three models (an "ablation": same data, same splits, same learner,
only the list of input columns changes)
    1. parcel_zoning_only    tabular parcel / zoning / terrain / flood columns
    2. imagery_only          AlphaEarth embedding columns only
    3. parcel_plus_imagery   both

    Model 2 is a "linear probe": a plain linear classifier trained on frozen
    embeddings from a foundation model. We never retrain the foundation
    model. If a simple linear model can predict development from the
    embeddings, the embeddings already contain that information.

Why the evaluation looks the way it does
    * SPATIAL BLOCK CROSS-VALIDATION is the headline number. Parcels close
      together resemble each other (spatial autocorrelation, also called
      Tobler's first law of geography). If neighbors land on both sides of a
      random train/test split, the model is graded partly on parcels it has
      effectively already seen, and the score is inflated. So we cut the
      city into square blocks and hold out WHOLE blocks at a time.
    * RANDOM K-fold is run too, only to show how big that inflation is.
    * A PLACEBO test swaps the imagery columns for junk of the same shape.
      If junk "helps" as much as imagery, then imagery was not really helping.
    * Every preprocessing step that learns from data (filling gaps, scaling,
      one-hot categories, PCA) lives inside an sklearn Pipeline, so it is
      fitted on the training fold only. Fitting it on all rows would leak
      information about the test fold into training ("data leakage").

Usage
    python train_models.py --data features.csv --outdir results
    python train_models.py --data features.csv --outdir results \
        --robustness --imagery-breakdown --tune-C

Outputs (in --outdir)
    summary.csv, metrics_by_fold.csv, comparisons.csv, comparison.txt,
    oof_predictions.csv, coefficients.csv, run_config.json,
    auc_by_model.png, risk_map.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# =============================================================================
# CONFIGURATION -- edit these lists to match your real column names.
# Every list can also be overridden from the command line (see --help).
# =============================================================================
ID_COL = "parcel_id"
LABEL_COL = "developed"          # 1 = new-construction permit 2021-2025, else 0
X_COL, Y_COL = "x", "y"          # parcel centroid, ideally in meters (EPSG:32614)

# Tabular features. ALL of these must describe 2020 or earlier.
TABULAR_NUMERIC = [
    "lot_area",                  # square meters
    "in_floodplain",             # 0/1
    "slope",                     # percent
    "elevation",                 # meters
    "dist_to_road",              # meters
    "improvement_value_ratio",   # improvement value / total value, 2020 roll
]
TABULAR_CATEGORICAL = [
    "zoning_class",
    "land_cover_class",
]
# Strongly right-skewed columns. A log transform keeps a handful of huge lots
# from dominating a linear model. log(1 + x) needs no fitting, so it is safe.
LOG_COLS = ["lot_area", "dist_to_road"]

# Imagery features (AlphaEarth). Columns are found by prefix.
PARCEL_EMB_PREFIX = "emb_A"      # emb_A00..emb_A63: mean inside the parcel
NBR_EMB_PREFIX = "nbr_A"         # nbr_A00..nbr_A63: mean in the 250 m ring
CHANGE_COLS = ["emb_change", "nbr_change"]   # 2018 -> 2020 change
CLAY_PREFIX = "clay_"            # optional Clay columns, used with --include-clay

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]  # values tried by --tune-C
MAIN_MODELS = ["parcel_zoning_only", "imagery_only", "parcel_plus_imagery"]
# =============================================================================

# Column-name fragments that suggest a feature describes the FUTURE (after the
# 2020 cutoff) or is the label in disguise. Used by the leakage screen.
SUSPICIOUS_NAME = re.compile(
    r"(202[1-9])|permit|year_?built|yr_?b(ui)?lt|complet|label|target|developed",
    re.IGNORECASE,
)


def safe_log1p(a):
    """log(1 + x), with negatives clipped to 0 so bad values cannot crash it."""
    return np.log1p(np.clip(a, 0, None))


# -----------------------------------------------------------------------------
# Loading and checking the data
# -----------------------------------------------------------------------------
def load_table(path: Path, args) -> pd.DataFrame:
    """Read a CSV (needs x/y columns) or a GeoJSON/GeoPackage (needs geopandas)."""
    if path.suffix.lower() in {".geojson", ".json", ".gpkg", ".shp"}:
        try:
            import geopandas as gpd
        except ImportError:
            sys.exit("Reading a spatial file needs geopandas. Install it, or "
                     "export a CSV that has x and y centroid columns.")
        gdf = gpd.read_file(path)
        # Centroids must be computed in a projected CRS (meters), not degrees.
        cent = gdf.geometry.to_crs(epsg=32614).centroid
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        df[args.x_col], df[args.y_col] = cent.x.to_numpy(), cent.y.to_numpy()
        return df
    return pd.read_csv(path, dtype={args.id_col: str}, low_memory=False)


def fail_missing(missing, df, what):
    cols = [c for c in df.columns
            if not c.startswith((PARCEL_EMB_PREFIX, NBR_EMB_PREFIX, CLAY_PREFIX))]
    sys.exit(
        f"\nERROR: missing {what}: {missing}\n"
        f"Non-embedding columns found in the file: {cols}\n"
        "Fix the file, edit the CONFIGURATION block at the top of "
        "train_models.py, or pass --numeric-cols / --categorical-cols."
    )


def prepare(df: pd.DataFrame, args):
    """Validate columns, clean types, and return (df, column lists)."""
    core = [args.id_col, args.label_col, args.x_col, args.y_col]
    missing = [c for c in core if c not in df.columns]
    if missing:
        fail_missing(missing, df, "required column(s)")

    numeric = list(args.numeric_cols)
    categorical = list(args.categorical_cols)
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        fail_missing(missing, df, "tabular feature column(s)")

    emb = sorted(c for c in df.columns if c.startswith(PARCEL_EMB_PREFIX))
    nbr = sorted(c for c in df.columns if c.startswith(NBR_EMB_PREFIX))
    chg = [c for c in CHANGE_COLS if c in df.columns]
    clay = sorted(c for c in df.columns if c.startswith(CLAY_PREFIX))
    if not emb:
        fail_missing([f"{PARCEL_EMB_PREFIX}00 ... {PARCEL_EMB_PREFIX}63"], df,
                     "imagery column(s)")
    if not nbr:
        print(f"NOTE: no '{NBR_EMB_PREFIX}*' neighborhood columns found; continuing without them.")
    if not chg:
        print(f"NOTE: no change columns {CHANGE_COLS} found; continuing without them.")

    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates(subset=args.id_col, keep="first")
    if len(df) < n0:
        print(f"WARNING: dropped {n0 - len(df)} rows with a duplicate {args.id_col}.")

    for c in numeric + emb + nbr + chg + clay + [args.x_col, args.y_col, args.label_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n1 = len(df)
    df = df.dropna(subset=[args.label_col, args.x_col, args.y_col])
    if len(df) < n1:
        print(f"WARNING: dropped {n1 - len(df)} rows with a missing label or coordinates.")
    labels = set(df[args.label_col].unique())
    if not labels <= {0, 1} or len(labels) < 2:
        sys.exit(f"ERROR: '{args.label_col}' must contain both 0 and 1 and nothing else. Found: {sorted(labels)}")
    df[args.label_col] = df[args.label_col].astype(int)

    # Missing categories become their own "missing" category. This uses no
    # information from other rows, so it cannot leak.
    for c in categorical:
        df[c] = df[c].astype(object).where(df[c].notna(), "missing").astype(str)

    # Block size is in the units of x/y. If the coordinates look like
    # longitude/latitude (degrees), convert them to approximate meters.
    x, y = df[args.x_col].to_numpy(), df[args.y_col].to_numpy()
    if np.abs(x).max() <= 180 and np.abs(y).max() <= 90:
        print("NOTE: x/y look like longitude/latitude. Converting to approximate "
              "meters for the spatial blocks. Projected coordinates (EPSG:32614) are better.")
        lat0 = np.deg2rad(y.mean())
        df[args.x_col] = (x - x.mean()) * 111_320 * np.cos(lat0)
        df[args.y_col] = (y - y.mean()) * 110_540

    cols = dict(numeric=numeric, categorical=categorical, emb=emb, nbr=nbr,
                chg=chg, clay=clay)
    return df.reset_index(drop=True), cols


def leakage_screen(df, cols, args):
    """Flag feature columns that look like the answer in disguise.

    This is a tripwire, not a proof of safety. It catches two things:
      * a single column that separates developed from undeveloped parcels
        almost perfectly by itself (no honest 2020 feature does that), and
      * column names that hint at post-2020 information.
    The real protection is knowing where every column came from and what
    year it describes.
    """
    y = df[args.label_col].to_numpy()
    flags = []
    numeric_like = cols["numeric"] + cols["emb"] + cols["nbr"] + cols["chg"] + cols["clay"]
    for c in numeric_like:
        v = df[c]
        if v.notna().sum() < 10 or v.nunique() < 2:
            continue
        auc = roc_auc_score(y, v.fillna(v.median()))
        sep = max(auc, 1 - auc)
        if sep >= args.leak_auc_threshold:
            flags.append(f"'{c}' alone separates the classes with AUC {sep:.3f} "
                         f"(threshold {args.leak_auc_threshold}). Is it derived from the label or from post-2020 data?")
    base = y.mean()
    for c in cols["categorical"]:
        g = df.groupby(c)[args.label_col].agg(["mean", "size"])
        pure = g[(g["size"] >= 30) & ((g["mean"] >= 0.95) | ((g["mean"] <= 0.0) & (base > 0.3)))]
        for cat, row in pure.iterrows():
            flags.append(f"category '{cat}' of '{c}' is {row['mean']:.0%} developed "
                         f"across {int(row['size'])} parcels. Check what that category means.")
    for c in cols["numeric"] + cols["categorical"]:
        if SUSPICIOUS_NAME.search(c):
            flags.append(f"the name '{c}' suggests post-2020 or label-derived information. Confirm it describes 2020 or earlier.")
    return flags


# -----------------------------------------------------------------------------
# Cross-validation folds
# -----------------------------------------------------------------------------
def make_blocks(x, y, block_size):
    """Assign each parcel to a square grid cell (a "block") by its centroid."""
    bx = np.floor((x - x.min()) / block_size).astype(int)
    by = np.floor((y - y.min()) / block_size).astype(int)
    return bx * 100_000 + by   # one integer id per grid cell


def spatial_folds(y, blocks, n_folds, seed):
    """Hold out whole blocks: all parcels in a block share a fold."""
    n_blocks = len(np.unique(blocks))
    if n_blocks < n_folds:
        sys.exit(f"ERROR: only {n_blocks} spatial blocks for {n_folds} folds. "
                 "Use a smaller --block-size or fewer --n-folds.")
    try:  # shuffle + seed needs scikit-learn >= 1.6
        gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    except TypeError:
        gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(np.zeros(len(y)), y, groups=blocks))


def random_folds(y, n_folds, seed):
    """Ordinary shuffled K-fold (stratified so every fold has positives)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def feature_sets(cols, args):
    """Which columns each model is allowed to see."""
    imagery = cols["emb"] + cols["nbr"] + cols["chg"]
    tab = dict(numeric=cols["numeric"], categorical=cols["categorical"])
    none = dict(numeric=[], categorical=[])
    sets = {
        "parcel_zoning_only": dict(**tab, imagery=[], clay=[]),
        "imagery_only": dict(**none, imagery=imagery, clay=[]),
        "parcel_plus_imagery": dict(**tab, imagery=imagery, clay=[]),
    }
    if args.imagery_breakdown:   # which part of the imagery carries the signal?
        if cols["emb"]:
            sets["parcel_plus_parcel_emb"] = dict(**tab, imagery=cols["emb"], clay=[])
        if cols["nbr"]:
            sets["parcel_plus_neighborhood_emb"] = dict(**tab, imagery=cols["nbr"], clay=[])
        if cols["chg"]:
            sets["parcel_plus_change"] = dict(**tab, imagery=cols["chg"], clay=[])
    if args.include_clay:
        if cols["clay"]:
            sets["parcel_plus_clay"] = dict(**tab, imagery=[], clay=cols["clay"])
        else:
            print(f"NOTE: --include-clay was set but no '{CLAY_PREFIX}*' columns exist; skipping the Clay model.")
    return sets


def make_learner(name, args):
    if name == "logreg":
        # L2-regularized logistic regression (L2 is scikit-learn's default
        # penalty). class_weight="balanced" makes the rare "developed" class
        # count as much as the common class. A side effect: predicted
        # probabilities come out too high, so read them as a RANKING score,
        # not as a literal chance of development.
        return LogisticRegression(C=args.C, class_weight="balanced", max_iter=5000)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=args.seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=args.seed, n_jobs=-1)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced", random_state=args.seed, verbose=-1)
    raise ValueError(name)


def embedding_steps(n_pca, seed):
    """Fill gaps, standardize, and optionally compress with PCA.

    PCA (principal component analysis) squeezes many correlated columns into
    a few summary columns. It is fitted on the training fold only, because
    this whole list of steps sits inside the Pipeline.
    """
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("scale", StandardScaler())]
    if n_pca and n_pca > 0:
        steps += [("pca", PCA(n_components=n_pca, random_state=seed)),
                  ("rescale", StandardScaler())]
    return Pipeline(steps)


def build_pipeline(spec, learner, args):
    """Preprocessing + classifier as ONE object, so nothing is fitted outside
    the training fold."""
    plain = [c for c in spec["numeric"] if c not in args.log_cols]
    logged = [c for c in spec["numeric"] if c in args.log_cols]
    parts = []
    if plain:
        parts.append(("tab_num", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]), plain))
    if logged:
        parts.append(("tab_log", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("log", FunctionTransformer(safe_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler())]), logged))
    if spec["categorical"]:
        # One-hot: one 0/1 column per category. Categories seen fewer than
        # --min-category-count times in the training fold are pooled, and a
        # category that only shows up in the test fold is treated as "rare".
        parts.append(("tab_cat", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=args.min_category_count, sparse_output=False),
            spec["categorical"]))
    if spec["imagery"]:
        n = min(args.pca_components, len(spec["imagery"])) if args.pca_components else 0
        parts.append(("img", embedding_steps(n, args.seed), spec["imagery"]))
    if spec["clay"]:
        n = min(args.clay_pca_components, len(spec["clay"])) if args.clay_pca_components else 0
        parts.append(("clay", embedding_steps(n, args.seed), spec["clay"]))
    prep = ColumnTransformer(parts, remainder="drop")
    return Pipeline([("prep", prep), ("clf", make_learner(learner, args))])


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def fold_metrics(y_true, score, top_frac):
    """Metrics for one held-out fold.

    roc_auc   chance that a random developed parcel outranks a random
              undeveloped one. 0.5 = coin flip, 1.0 = perfect.
    pr_auc    average precision. More informative than ROC-AUC when positives
              are rare. Its "coin flip" value is the base rate, not 0.5.
    top-K     if a planner could only look at the top K% of parcels by score:
              precision = share of those that really were developed,
              recall = share of all developed parcels that were caught,
              lift = precision / base rate (1.0 = no better than random).
    """
    base = float(y_true.mean())
    out = dict(n_test=len(y_true), n_pos_test=int(y_true.sum()), base_rate=base)
    if 0 < y_true.sum() < len(y_true):
        out["roc_auc"] = roc_auc_score(y_true, score)
        out["pr_auc"] = average_precision_score(y_true, score)
    else:
        out["roc_auc"] = out["pr_auc"] = np.nan
    n_top = max(1, int(np.ceil(top_frac * len(y_true))))
    top = np.argsort(-score, kind="stable")[:n_top]
    hits = y_true[top].sum()
    out["precision_at_k"] = hits / n_top
    out["recall_at_k"] = hits / y_true.sum() if y_true.sum() else np.nan
    out["lift_at_k"] = out["precision_at_k"] / base if base > 0 else np.nan
    return out


def run_cv(df, y, spec, folds, scheme, blocks, learner, args):
    """Train on k-1 folds, score the held-out fold, repeat.

    Returns per-fold metric rows and the "out-of-fold" (OOF) scores: for each
    parcel, the score from the one model that never saw it during training.
    """
    used = spec["numeric"] + spec["categorical"] + spec["imagery"] + spec["clay"]
    X = df[used]
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, te) in enumerate(folds):
        model = build_pipeline(spec, learner, args)
        if args.tune_C and learner == "logreg":
            # Pick the regularization strength C using ONLY the training
            # fold. The inner splits follow the same rule as the outer ones
            # (whole blocks for spatial CV), otherwise tuning would cheat.
            if scheme == "spatial":
                n_inner = min(3, len(np.unique(blocks[tr])))
                inner = list(GroupKFold(n_splits=n_inner).split(tr, y[tr], groups=blocks[tr]))
            else:
                inner = list(StratifiedKFold(3, shuffle=True, random_state=args.seed).split(tr, y[tr]))
            model = GridSearchCV(model, {"clf__C": C_GRID}, scoring="roc_auc", cv=inner)
        model.fit(X.iloc[tr], y[tr])
        score = model.predict_proba(X.iloc[te])[:, 1]
        oof[te] = score
        row = dict(fold=k, n_train=len(tr), **fold_metrics(y[te], score, args.top_frac))
        if args.tune_C and learner == "logreg":
            row["chosen_C"] = model.best_params_["clf__C"]
        if np.isnan(row["roc_auc"]):
            print(f"WARNING: fold {k} ({scheme}) has only one class in the test set; AUC is undefined there.")
        rows.append(row)
    return rows, oof


def placebo_frame(df, imagery_cols, kind, seed):
    """Return a copy of df whose imagery columns are junk.

    "shuffled": every parcel gets some OTHER parcel's real imagery row. The
                numbers keep their realistic structure but no longer belong
                to the right parcel.
    "noise":    pure random numbers.
    Same number of columns as the real thing, so a score gain from "just
    having more columns" would show up here too.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    block = df[imagery_cols].to_numpy()
    if kind == "shuffled":
        out[imagery_cols] = block[rng.permutation(len(df))]
    else:
        out[imagery_cols] = rng.standard_normal(block.shape)
    return out


def paired_fold_test(a, b):
    """Compare two models fold by fold (same folds, so the scores are paired).

    With only ~5 folds these tests are weak, and folds share training data so
    the p-values are approximate. Treat them as supporting evidence. A
    Wilcoxon test on 5 folds cannot give p below 0.0625.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    out = dict(n_folds=int(ok.sum()), mean_diff=d.mean() if len(d) else np.nan,
               std_diff=d.std(ddof=1) if len(d) > 1 else np.nan,
               folds_a_wins=int((d > 0).sum()), t_p=np.nan, wilcoxon_p=np.nan)
    if len(d) > 1 and d.std(ddof=1) > 0:
        out["t_p"] = stats.ttest_rel(a[ok], b[ok]).pvalue
        try:
            out["wilcoxon_p"] = stats.wilcoxon(d).pvalue
        except ValueError:
            pass
    return out


def block_bootstrap_diff(y, score_a, score_b, blocks, n_boot, seed):
    """Confidence interval for (model A - model B) on the out-of-fold scores.

    An ordinary bootstrap resamples parcels as if they were independent.
    They are not: neighbors move together. So we resample whole BLOCKS with
    replacement, which keeps each neighborhood intact and gives an honestly
    wider interval.
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(blocks)
    members = {b: np.flatnonzero(blocks == b) for b in ids}
    d_auc, d_pr = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[b] for b in rng.choice(ids, len(ids), replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        d_auc.append(roc_auc_score(yy, score_a[idx]) - roc_auc_score(yy, score_b[idx]))
        d_pr.append(average_precision_score(yy, score_a[idx]) - average_precision_score(yy, score_b[idx]))
    lo_a, hi_a = np.percentile(d_auc, [2.5, 97.5])
    lo_p, hi_p = np.percentile(d_pr, [2.5, 97.5])
    return dict(
        pooled_auc_diff=roc_auc_score(y, score_a) - roc_auc_score(y, score_b),
        auc_ci_low=lo_a, auc_ci_high=hi_a,
        pooled_pr_auc_diff=average_precision_score(y, score_a) - average_precision_score(y, score_b),
        pr_ci_low=lo_p, pr_ci_high=hi_p)


def morans_i(values, xy, k, n_perm, seed):
    """Moran's I: do the model's ERRORS cluster in space?

    For every parcel, compare its error to the average error of its k
    nearest neighbors. I near 0 means errors are scattered at random. I well
    above 0 means the model is wrong in the same direction across whole
    areas, so some spatially organized driver is missing from the features.
    The p-value comes from shuffling the errors across parcels many times
    and counting how often chance alone gives an I this large.
    """
    z = values - values.mean()
    _, nn = cKDTree(xy).query(xy, k=k + 1)   # first neighbor is the parcel itself
    nn = nn[:, 1:]
    stat = lambda v: float((v * v[nn].mean(axis=1)).sum() / (v * v).sum())
    observed = stat(z)
    rng = np.random.default_rng(seed)
    perms = np.array([stat(rng.permutation(z)) for _ in range(n_perm)])
    p = (np.sum(perms >= observed) + 1) / (n_perm + 1)
    return observed, p


# -----------------------------------------------------------------------------
# Interpretation
# -----------------------------------------------------------------------------
def feature_family(name):
    """Map a transformed column name back to its feature family."""
    if name.startswith(("tab_num__", "tab_log__", "tab_cat__")):
        return "tabular (parcel/zoning)"
    if name.startswith("clay__"):
        return "Clay embedding"
    raw = name.split("__", 1)[-1]
    if raw in CHANGE_COLS:
        return "change 2018-2020"
    if raw.startswith(PARCEL_EMB_PREFIX):
        return "parcel embedding"
    if raw.startswith(NBR_EMB_PREFIX):
        return "neighborhood embedding"
    return "imagery (PCA components)"


def coefficient_report(df, y, spec, args):
    """Fit ONE logistic model on all rows, for interpretation only.

    (No score is computed from this fit, so using all rows is fine.)

    Single embedding dimensions have no human meaning ("A17" is not "trees"),
    so we do not interpret them one by one. We group them into families and
    ask how much each family moves the prediction: the standard deviation,
    across parcels, of that family's total contribution to the model's
    log-odds. A family with spread near 0 barely changes any prediction.
    """
    pipe = build_pipeline(spec, "logreg", args)
    used = spec["numeric"] + spec["categorical"] + spec["imagery"] + spec["clay"]
    pipe.fit(df[used], y)
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    Z = pipe.named_steps["prep"].transform(df[used])
    table = pd.DataFrame({"feature": names, "family": [feature_family(n) for n in names],
                          "coefficient": coef})
    fam_rows = []
    for fam, g in table.groupby("family"):
        contribution = Z[:, g.index] @ coef[g.index]
        fam_rows.append(dict(family=fam, n_features=len(g),
                             contribution_spread=float(contribution.std())))
    families = pd.DataFrame(fam_rows).sort_values("contribution_spread", ascending=False)
    return table, families


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]


def make_figures(summary, df, y, oof, outdir, stamp, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("NOTE: matplotlib not installed; skipping figures.")
        return
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})

    # ---- Figure 1: AUC per model, spatial vs random CV ----------------------
    s = summary[summary["learner"] == "logreg"]
    order = [m for m in s["model"].unique()]
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(order) + 1.9))
    for scheme, color, marker, dy in [("spatial", BLUE, "o", 0.16), ("random", ORANGE, "D", -0.16)]:
        part = s[s["cv_scheme"] == scheme].set_index("model")
        ys = [len(order) - 1 - order.index(m) + dy for m in part.index]
        ax.errorbar(part["roc_auc_mean"], ys, xerr=part["roc_auc_std"], fmt=marker,
                    color=color, ecolor=color, elinewidth=1.5, capsize=3, markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1,
                    label=f"{scheme} cross-validation" + (" (headline)" if scheme == "spatial" else " (optimistic)"))
        for xv, sd, yv in zip(part["roc_auc_mean"], part["roc_auc_std"], ys):
            ax.text(xv + sd + 0.006, yv, f"{xv:.3f}", va="center", fontsize=9, color=INK2)
    ax.axvline(0.5, color=AXIS, lw=1)
    ax.text(0.5, len(order) - 0.45, " coin flip", fontsize=8.5, color=MUTED, va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order[::-1])
    ax.set_ylim(-0.6, len(order) - 0.2)
    lo = min(0.48, float((s["roc_auc_mean"] - s["roc_auc_std"]).min()) - 0.02)
    ax.set_xlim(lo, min(1.0, float((s["roc_auc_mean"] + s["roc_auc_std"]).max()) + 0.06))
    ax.set_xlabel("ROC-AUC (mean across folds; bars show ±1 standard deviation)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, labelcolor=INK2)
    ax.set_title(f"How well each model ranks parcels{stamp}", loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(outdir / "auc_by_model.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: map of out-of-fold scores next to what happened ----------
    key = ("spatial", "logreg", "parcel_plus_imagery")
    if key not in oof:
        return
    x, yy = df[args.x_col].to_numpy() / 1000, df[args.y_col].to_numpy() / 1000
    # class_weight="balanced" inflates raw probabilities, so show each parcel's
    # percentile rank instead: 99 = scored higher than 99% of parcels.
    rank = stats.rankdata(oof[key]) / len(y) * 100
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    size = float(np.clip(40_000 / len(y), 0.4, 6))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharex=True, sharey=True)
    # Draw parcels in random order so neither high nor low scores hide the other.
    order_idx = np.random.default_rng(args.seed).permutation(len(y))
    sc = axes[0].scatter(x[order_idx], yy[order_idx], c=rank[order_idx], cmap=cmap,
                         s=size, linewidths=0, vmin=0, vmax=100)
    axes[0].set_title("Predicted: development score (percentile rank)", loc="left", fontsize=11)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("percentile rank of out-of-fold score", color=INK2)
    axes[1].scatter(x[y == 0], yy[y == 0], c=AXIS, s=size, linewidths=0, label="not developed")
    axes[1].scatter(x[y == 1], yy[y == 1], c=BLUE_RAMP[3], s=size * 1.6, linewidths=0,
                    label="developed (new-construction permit)")
    axes[1].set_title("Actual: what happened 2021-2025", loc="left", fontsize=11)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                   ncol=2, markerscale=4, labelcolor=INK2)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("easting (km)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
    axes[0].set_ylabel("northing (km)")
    fig.suptitle(f"Model 3 (parcel + imagery), scored on held-out spatial blocks{stamp}",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "risk_map.png", dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Plain-English readout
# -----------------------------------------------------------------------------
def fmt_p(p):
    return "n/a" if p is None or np.isnan(p) else f"{p:.3f}"


def write_readout(path, ctx):
    a = ctx["args"]
    S = ctx["summary"].set_index(["cv_scheme", "learner", "model"])
    L = []
    add = L.append
    if ctx["synthetic"]:
        add("#" * 74)
        add("#  SYNTHETIC DATA. These numbers come from make_synthetic_data.py.")
        add("#  They test the code. They say NOTHING about real Austin parcels.")
        add("#" * 74)
        add("")
    add("THREE-MODEL COMPARISON")
    add(f"data file: {a.data}")
    add(f"parcels: {ctx['n']:,}   developed: {ctx['n_pos']:,} ({ctx['n_pos'] / ctx['n']:.1%})   "
        f"spatial blocks: {ctx['n_blocks']} of {a.block_size:g} m   folds: {a.n_folds}")
    add("")

    if ctx["leak_flags"]:
        add("!!! POSSIBLE DATA LEAKAGE. Resolve these before trusting any number below:")
        for f in ctx["leak_flags"]:
            add(f"  - {f}")
        add("")

    add("1. HEADLINE: spatial block cross-validation, logistic regression")
    add(f"   {'model':<30}{'ROC-AUC':>16}{'PR-AUC':>16}{'top-' + format(a.top_frac, '.0%') + ' lift':>14}")
    for m in ctx["order"]:
        r = S.loc[("spatial", "logreg", m)]
        add(f"   {m:<30}{r['roc_auc_mean']:>9.3f} ±{r['roc_auc_std']:.3f}"
            f"{r['pr_auc_mean']:>9.3f} ±{r['pr_auc_std']:.3f}{r['lift_at_k_mean']:>13.2f}x")
    add(f"   (± is the standard deviation across folds. PR-AUC of a random guess = base rate = {ctx['n_pos'] / ctx['n']:.3f}.)")
    add("")

    c = ctx["comparisons"].set_index(["model_a", "model_b"])
    main = c.loc[("parcel_plus_imagery", "parcel_zoning_only")]
    add("2. DOES IMAGERY HELP?  parcel_plus_imagery minus parcel_zoning_only (spatial CV)")
    add(f"   mean AUC difference across folds: {main['mean_diff']:+.4f}  "
        f"(imagery model ahead in {int(main['folds_a_wins'])} of {int(main['n_folds'])} folds)")
    add(f"   paired t-test p = {fmt_p(main['t_p'])}   Wilcoxon p = {fmt_p(main['wilcoxon_p'])}   "
        "(few folds: weak tests, approximate p-values)")
    add(f"   AUC difference on pooled out-of-fold scores: {main['pooled_auc_diff']:+.4f}, "
        f"95% block-bootstrap CI [{main['auc_ci_low']:+.4f}, {main['auc_ci_high']:+.4f}]")
    add(f"   PR-AUC difference on pooled out-of-fold scores: {main['pooled_pr_auc_diff']:+.4f}, "
        f"95% CI [{main['pr_ci_low']:+.4f}, {main['pr_ci_high']:+.4f}]")

    placebo_best = max((c.loc[(p, "parcel_zoning_only")]["mean_diff"] for p in ctx["placebos"]),
                       default=np.nan)
    if main["auc_ci_low"] > 0:
        verdict = ("VERDICT: imagery features improved ranking beyond parcel and zoning data. "
                   "The whole confidence interval is above zero.")
        if not np.isnan(placebo_best) and placebo_best >= 0.5 * main["mean_diff"]:
            verdict += (" CAUTION: a placebo gained at least half as much, so part of the "
                        "gain may be an artifact. Investigate before claiming it.")
    elif main["auc_ci_high"] < 0:
        verdict = ("VERDICT: adding imagery features made ranking WORSE. The whole confidence "
                   "interval is below zero. Likely overfitting to many extra columns; try "
                   "--pca-components or a smaller --C. Report it as a negative result.")
    else:
        verdict = ("VERDICT: no reliable evidence that imagery features help. The confidence "
                   "interval includes zero. Report this honestly as a null result.")
    add("   " + verdict)
    add("")

    add("3. PLACEBO TEST (imagery columns replaced by junk of the same shape, spatial CV)")
    for p in ctx["placebos"]:
        r = c.loc[(p, "parcel_zoning_only")]
        add(f"   {p:<30} AUC {S.loc[('spatial', 'logreg', p)]['roc_auc_mean']:.3f}   "
            f"vs parcel_zoning_only: {r['mean_diff']:+.4f}   95% CI [{r['auc_ci_low']:+.4f}, {r['auc_ci_high']:+.4f}]")
    add("   Expected: at or slightly below zero. Junk columns should never help. If a placebo")
    add("   shows a clear gain, something is leaking and the imagery result cannot be trusted.")
    add("")

    add("4. SPATIAL vs RANDOM cross-validation (mean ROC-AUC, logistic regression)")
    add(f"   {'model':<30}{'spatial':>10}{'random':>10}{'inflation':>12}")
    for m in ctx["order"]:
        sp = S.loc[("spatial", "logreg", m)]["roc_auc_mean"]
        rd = S.loc[("random", "logreg", m)]["roc_auc_mean"]
        add(f"   {m:<30}{sp:>10.3f}{rd:>10.3f}{rd - sp:>+12.3f}")
    r = c.loc[("parcel_plus_imagery", "parcel_zoning_only")]
    add(f"   Imagery gain under RANDOM CV would have been {ctx['random_gain']:+.4f}; "
        f"under spatial CV it is {r['mean_diff']:+.4f}.")
    add("   Random CV lets a model recognize a test parcel's neighbors from training. The")
    add("   spatial number is the one to report. The gap shows why.")
    add("")

    add("5. DO THE ERRORS CLUSTER IN SPACE?  Moran's I on out-of-fold residuals")
    for m, (i, p) in ctx["moran"].items():
        add(f"   {m:<30} I = {i:+.3f}   permutation p = {p:.3f}")
    add("   I near 0: errors look random. I clearly above 0: the model is wrong in the same")
    add("   direction across whole areas, so a spatially organized driver is still missing.")
    add("")

    add("6. WHICH FEATURE FAMILIES MOVE THE PREDICTIONS?  (model 3, fitted on all rows)")
    for _, r in ctx["families"].iterrows():
        add(f"   {r['family']:<28} {int(r['n_features']):>4} features   contribution spread {r['contribution_spread']:.3f}")
    add("   Spread = std. dev. across parcels of that family's total push on the log-odds.")
    add("   Single embedding dimensions are not human-interpretable. Interpret imagery by")
    add("   family and by the map (risk_map.png), never as 'band A17 means X'.")
    top = ctx["coef_table"]
    top = top[top["family"] == "tabular (parcel/zoning)"].copy()
    top["abs"] = top["coefficient"].abs()
    add("   Largest tabular coefficients (standardized; + raises development odds):")
    for _, r in top.sort_values("abs", ascending=False).head(8).iterrows():
        add(f"     {r['feature'].split('__', 1)[-1]:<38}{r['coefficient']:+.3f}")
    add("")

    n = 7
    if ctx["tiny"]:
        add(f"{n}. SMALL PARCELS (is_tiny_parcel = 1: under ~3 pixels, embedding is a fallback sample)")
        for line in ctx["tiny"]:
            add("   " + line)
        add("")
        n += 1
    extra = [m for m in ctx["all_models"] if m not in ctx["order"]]
    if extra:
        add(f"{n}. EXTRA FEATURE SETS (spatial CV, logistic regression)")
        for m in extra:
            r = S.loc[("spatial", "logreg", m)]
            d = c.loc[(m, "parcel_zoning_only")]
            add(f"   {m:<30} AUC {r['roc_auc_mean']:.3f} ±{r['roc_auc_std']:.3f}   "
                f"vs parcel_zoning_only {d['mean_diff']:+.4f}  CI [{d['auc_ci_low']:+.4f}, {d['auc_ci_high']:+.4f}]")
        add("")
        n += 1
    robust = ctx["summary"][ctx["summary"]["learner"] != "logreg"]
    if len(robust):
        add(f"{n}. ROBUSTNESS: other learners on the same three feature sets (spatial CV)")
        for _, r in robust.iterrows():
            add(f"   {r['learner']:<15}{r['model']:<28} AUC {r['roc_auc_mean']:.3f} ±{r['roc_auc_std']:.3f}   "
                f"PR-AUC {r['pr_auc_mean']:.3f}")
        add("   If the ordering of the three models matches the logistic results, the conclusion")
        add("   does not depend on the choice of learner.")
        add("")

    add("NOTES")
    add("- Scores come from class-weighted models, so treat them as rankings, not as literal")
    add("  probabilities of development.")
    add("- 'Pooled' differences score all out-of-fold predictions together; fold means score")
    add("  each fold separately. Small disagreements between the two are normal.")
    add("- The label is a new-construction PERMIT. It measures development likelihood, not")
    add("  pure physical suitability, and a permit is not a completed building.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Train and compare parcel-only, imagery-only and parcel+imagery models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", required=True, help="Feature table: CSV with x/y columns, or GeoJSON.")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--id-col", default=ID_COL)
    ap.add_argument("--label-col", default=LABEL_COL)
    ap.add_argument("--x-col", default=X_COL)
    ap.add_argument("--y-col", default=Y_COL)
    ap.add_argument("--numeric-cols", nargs="+", default=TABULAR_NUMERIC)
    ap.add_argument("--categorical-cols", nargs="*", default=TABULAR_CATEGORICAL)
    ap.add_argument("--log-cols", nargs="*", default=LOG_COLS)
    ap.add_argument("--min-category-count", type=int, default=20,
                    help="Categories rarer than this (in a training fold) are pooled.")
    ap.add_argument("--block-size", type=float, default=2000.0,
                    help="Side of a spatial CV block, in the units of x/y (meters).")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0,
                    help="Inverse regularization strength for logistic regression (smaller = stronger).")
    ap.add_argument("--tune-C", action="store_true",
                    help=f"Choose C from {C_GRID} by inner CV on each training fold.")
    ap.add_argument("--pca-components", type=int, default=0,
                    help="Compress imagery columns to this many PCA components (0 = off).")
    ap.add_argument("--top-frac", type=float, default=0.10,
                    help="Top share of parcels used for precision/recall/lift.")
    ap.add_argument("--robustness", action="store_true",
                    help="Also run random forest (and XGBoost/LightGBM if installed).")
    ap.add_argument("--imagery-breakdown", action="store_true",
                    help="Also run parcel + each imagery family separately.")
    ap.add_argument("--include-clay", action="store_true",
                    help="Add a parcel_plus_clay model if clay_* columns exist.")
    ap.add_argument("--clay-pca-components", type=int, default=32,
                    help="PCA components for Clay columns (1,024 raw dims need compressing).")
    ap.add_argument("--n-boot", type=int, default=1000, help="Block-bootstrap resamples.")
    ap.add_argument("--moran-k", type=int, default=8, help="Neighbors for Moran's I.")
    ap.add_argument("--moran-perms", type=int, default=199)
    ap.add_argument("--leak-auc-threshold", type=float, default=0.90,
                    help="Flag any single feature whose own AUC reaches this.")
    ap.add_argument("--strict-leakage", action="store_true",
                    help="Stop instead of warning when the leakage screen flags something.")
    ap.add_argument("--no-figures", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, cols = prepare(load_table(Path(args.data), args), args)
    y = df[args.label_col].to_numpy()
    synthetic = "is_synthetic" in df.columns and bool((df["is_synthetic"] == 1).any())
    data_tag = "SYNTHETIC" if synthetic else "real"
    stamp = "\nSYNTHETIC DATA: tests the code, not a real result" if synthetic else ""
    print(f"Loaded {len(df):,} parcels, {int(y.sum()):,} developed ({y.mean():.1%}). Data: {data_tag}")

    # ---- Leakage screen ------------------------------------------------------
    leak_flags = leakage_screen(df, cols, args)
    if leak_flags:
        print("\n" + "!" * 74 + "\nPOSSIBLE DATA LEAKAGE:")
        for f in leak_flags:
            print("  - " + f)
        print("!" * 74 + "\n")
        if args.strict_leakage:
            sys.exit("Stopping because --strict-leakage is set. Fix or remove the flagged columns.")

    # ---- Folds (built once, reused by every model so comparisons are paired) -
    blocks = make_blocks(df[args.x_col].to_numpy(), df[args.y_col].to_numpy(), args.block_size)
    folds = {"spatial": spatial_folds(y, blocks, args.n_folds, args.seed),
             "random": random_folds(y, args.n_folds, args.seed)}
    fold_id = {}
    for scheme, fl in folds.items():
        fid = np.empty(len(y), int)
        for k, (_, te) in enumerate(fl):
            fid[te] = k
        fold_id[scheme] = fid
    print(f"Spatial CV: {len(np.unique(blocks))} blocks of {args.block_size:g} m in {args.n_folds} folds.")

    sets = feature_sets(cols, args)
    rows, oof = [], {}

    def record(scheme, learner, name, spec, frame):
        r, p = run_cv(frame, y, spec, folds[scheme], scheme, blocks, learner, args)
        for row in r:
            rows.append(dict(cv_scheme=scheme, learner=learner, model=name, **row))
        oof[(scheme, learner, name)] = p
        auc = np.nanmean([x["roc_auc"] for x in r])
        print(f"  {scheme:<8}{learner:<15}{name:<30} mean AUC {auc:.3f}")

    print("\nTraining (logistic regression):")
    for scheme in ("spatial", "random"):
        for name, spec in sets.items():
            record(scheme, "logreg", name, spec, df)

    # ---- Placebos: same model-3 recipe, junk imagery -------------------------
    spec3 = sets["parcel_plus_imagery"]
    placebos = []
    for kind in ("shuffled", "noise"):
        name = f"placebo_{kind}_imagery"
        record("spatial", "logreg", name, spec3,
               placebo_frame(df, spec3["imagery"], kind, args.seed + 1))
        placebos.append(name)

    # ---- Optional robustness learners ----------------------------------------
    if args.robustness:
        learners = ["random_forest"]
        for lib, name in (("xgboost", "xgboost"), ("lightgbm", "lightgbm")):
            try:
                __import__(lib)
                learners.append(name)
            except ImportError:
                print(f"NOTE: {lib} not installed; skipping it.")
        print("\nRobustness check (spatial CV):")
        for learner in learners:
            for name in MAIN_MODELS:
                record("spatial", learner, name, sets[name], df)

    # ---- Tables --------------------------------------------------------------
    by_fold = pd.DataFrame(rows)
    by_fold.insert(0, "data", data_tag)
    metric_cols = ["roc_auc", "pr_auc", "precision_at_k", "recall_at_k", "lift_at_k"]
    summary = (by_fold.groupby(["cv_scheme", "learner", "model"], sort=False)[metric_cols]
               .agg(["mean", "std"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary["pooled_oof_roc_auc"] = [roc_auc_score(y, oof[(r.cv_scheme, r.learner, r.model)])
                                     for r in summary.itertuples()]
    summary.insert(0, "data", data_tag)

    # ---- Paired comparisons against the parcel-only baseline -----------------
    def fold_aucs(scheme, model):
        m = by_fold[(by_fold.cv_scheme == scheme) & (by_fold.learner == "logreg") & (by_fold.model == model)]
        return m.sort_values("fold")["roc_auc"].to_numpy()

    comps = []
    base = "parcel_zoning_only"
    for name in [m for m in list(sets) + placebos if m != base]:
        row = dict(data=data_tag, cv_scheme="spatial", model_a=name, model_b=base)
        row.update(paired_fold_test(fold_aucs("spatial", name), fold_aucs("spatial", base)))
        row.update(block_bootstrap_diff(y, oof[("spatial", "logreg", name)],
                                        oof[("spatial", "logreg", base)], blocks,
                                        args.n_boot, args.seed))
        comps.append(row)
    comparisons = pd.DataFrame(comps)
    random_gain = float(np.mean(fold_aucs("random", "parcel_plus_imagery")
                                - fold_aucs("random", base)))

    # ---- Moran's I on out-of-fold residuals ----------------------------------
    xy = df[[args.x_col, args.y_col]].to_numpy()
    moran = {m: morans_i(y - oof[("spatial", "logreg", m)], xy, args.moran_k,
                         args.moran_perms, args.seed) for m in MAIN_MODELS}

    # ---- Interpretation ------------------------------------------------------
    coef_table, families = coefficient_report(df, y, spec3, args)
    coef_table.insert(0, "data", data_tag)

    tiny_lines = []
    if "is_tiny_parcel" in df.columns:
        tiny = pd.to_numeric(df["is_tiny_parcel"], errors="coerce").fillna(0).to_numpy() == 1
        for label, mask in (("tiny", tiny), ("not tiny", ~tiny)):
            if mask.sum() >= 50 and 0 < y[mask].sum() < mask.sum():
                a1 = roc_auc_score(y[mask], oof[("spatial", "logreg", base)][mask])
                a3 = roc_auc_score(y[mask], oof[("spatial", "logreg", "parcel_plus_imagery")][mask])
                tiny_lines.append(f"{label:<9} n = {mask.sum():>6,}   parcel-only AUC {a1:.3f}   "
                                  f"parcel+imagery AUC {a3:.3f}   gain {a3 - a1:+.3f}")
        if tiny_lines:
            tiny_lines.append("If the gain is smaller for tiny parcels, 10 m pixels are too coarse for them. Say so.")
            tiny_lines.append("Compare the GAINS, not the raw AUCs: each group spans a narrow range of lot sizes,")
            tiny_lines.append("so within-group AUCs run lower than the headline.")

    # ---- Write everything ----------------------------------------------------
    by_fold.to_csv(outdir / "metrics_by_fold.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    comparisons.to_csv(outdir / "comparisons.csv", index=False)
    coef_table.to_csv(outdir / "coefficients.csv", index=False)

    oof_rows = []
    for (scheme, learner, name), p in oof.items():
        oof_rows.append(pd.DataFrame({
            "data": data_tag, args.id_col: df[args.id_col].to_numpy(), "cv_scheme": scheme,
            "learner": learner, "model": name, "fold": fold_id[scheme],
            "predicted_score": np.round(p, 5), "actual": y}))
    pd.concat(oof_rows).to_csv(outdir / "oof_predictions.csv", index=False)

    config = {k: v for k, v in vars(args).items()}
    config.update(data_kind=data_tag, n_parcels=int(len(y)), n_developed=int(y.sum()),
                  n_parcel_emb=len(cols["emb"]), n_nbr_emb=len(cols["nbr"]),
                  change_cols=cols["chg"], n_clay=len(cols["clay"]))
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    text = write_readout(outdir / "comparison.txt", dict(
        args=args, summary=summary, comparisons=comparisons, synthetic=synthetic,
        n=len(y), n_pos=int(y.sum()), n_blocks=len(np.unique(blocks)),
        order=MAIN_MODELS, all_models=list(sets), placebos=placebos,
        leak_flags=leak_flags, moran=moran, families=families, coef_table=coef_table,
        tiny=tiny_lines, random_gain=random_gain))
    print("\n" + text)

    if not args.no_figures:
        fig_summary = summary[summary["model"].isin(MAIN_MODELS + placebos)]
        make_figures(fig_summary, df, y, oof, outdir, stamp, args)
    print(f"\nAll outputs written to {outdir}/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
