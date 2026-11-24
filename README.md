# Site suitability model: pipeline and three-model comparison

GRG 327G / CS 378 group project (Gunner Johnson, Jay Patil, Palash Pawar).

**Research question.** Does adding imagery features from a geospatial foundation model help predict which Austin parcels get developed, compared to parcel and zoning data alone?

**Design in one paragraph.** One row per parcel. The label is `developed = 1` if the parcel got a new-construction permit in 2021–2025. Every feature describes 2020 or earlier. The pool is parcels that were vacant or underused in 2020, inside Austin city limits. Three models use the same data, the same splits and the same learner (logistic regression), and differ only in which columns they see: parcel and zoning columns, imagery columns, or both.

> **Status.** `train_models.py` and `make_synthetic_data.py` were run end to end on synthetic data. Nothing here has been run on real Austin data yet. `ee_extract_embeddings.py` and the new parts of `build_labels.py` are syntax-checked only. See [Limitations](#limitations).

## Files

| File | What it does | Tested? |
|---|---|---|
| `pull_permits.py` | Downloads Austin construction permits (the label source) | Not run here (no network) |
| `pull_parcels.py` | Downloads TCAD parcel polygons | Not run here (no network) |
| `build_labels.py` | Joins permits to parcels and writes the 0/1 label | Syntax only |
| `build_features.py` | Adds the tabular columns, applies the vacant/underused filter, joins embeddings, writes `features.csv` | Fake geometries only (`tests/`) |
| `ee_extract_embeddings.py` | Averages AlphaEarth embeddings over each parcel in Earth Engine | Syntax, plus unit tests of its math helpers |
| `train_models.py` | Trains and compares the three models | Run on synthetic data |
| `make_synthetic_data.py` | Builds fake data with a known answer, for testing | Run |
| `synthetic_demo/` | Outputs of the two synthetic test runs | Synthetic, not real results |
| `AI_ACKNOWLEDGMENT.md` | Text for citing AI use | |

## Setup

```bash
pip install pandas numpy scikit-learn scipy matplotlib geopandas earthengine-api
```

`train_models.py` needs only the first five. `geopandas` is needed by `build_labels.py` and `ee_extract_embeddings.py`. `xgboost` and `lightgbm` are optional and are used by `--robustness` if installed.

## Run the whole pipeline in Google Colab

`pipeline_colab.ipynb` walks through every real-data step in order, including the Earth Engine sign-in. Upload it at https://colab.research.google.com (File → Upload notebook). Data is kept in a Google Drive folder, not in this repository. The notebook has been syntax-checked only and has not been run.

## Try it first on fake data (2 minutes, no downloads)

```bash
python make_synthetic_data.py --out synthetic_demo/features_signal_on.csv
python train_models.py --data synthetic_demo/features_signal_on.csv --outdir synthetic_demo/signal_on

python make_synthetic_data.py --imagery-signal 0 --out synthetic_demo/features_signal_off.csv
python train_models.py --data synthetic_demo/features_signal_off.csv --outdir synthetic_demo/signal_off
```

Open `comparison.txt` in each output folder. The two CSVs are about 10 MB each and are not kept in this folder. The commands above recreate them exactly (fixed seed).

## The real pipeline, in order

```bash
# 1. Permits. Start at 2021-01-01 so the whole label window is covered.
python pull_permits.py --since 2021-01-01 --out permits_raw.csv

# 2. Parcels. Test pull first, read the printed field names, then the full pull.
python pull_parcels.py --out parcels_test.geojson --max-pages 3
python pull_parcels.py --out parcels_raw.geojson
#    For the final run you need the 2020 version of the parcels, not today's.

# 3. Labels. Location mode matches permits to parcels by where they are.
python build_labels.py --permits permits_raw.csv --parcels parcels_2020.geojson \
    --parcel-id-field <ID FIELD FROM STEP 2> \
    --label-mode location --cutoff-year 2020 --negative-sample-ratio 0 \
    --out labeled_parcels.geojson

# 4. Imagery features. Pilot first, then everything.
python ee_extract_embeddings.py --project <YOUR_GCP_PROJECT> \
    --parcels labeled_parcels.geojson --out embeddings_pilot.csv --pilot 50
python ee_extract_embeddings.py --project <YOUR_GCP_PROJECT> \
    --parcels labeled_parcels.geojson --out embeddings.csv --terrain-out terrain.csv
#    --terrain-out also writes elevation, slope (USGS 3DEP) and NLCD 2019 land cover.

# 5. Features. build_features.py adds the tabular columns, keeps only parcels
#    that were vacant or underused in 2020, and joins the embeddings.
#    Leave out any layer you do not have; the script prints what to do.
python build_features.py --labels labeled_parcels.geojson --parcel-year 2020 \
    --appraisal appraisal_2020.csv --appraisal-year 2020 \
    --appraisal-id-col <ROLL ID COLUMN> --labels-join-col <MATCHING PARCEL COLUMN> \
    --zoning zoning.geojson --floodplain flood_hazard.geojson --roads roads_2020.shp \
    --city-limits austin_full_purpose.geojson \
    --terrain terrain.csv --embeddings embeddings.csv --out features.csv

# 6. Train and compare.
python train_models.py --data features.csv --outdir results --tune-C --robustness --imagery-breakdown
```

## Column schema for `train_models.py`

One row per parcel. Names can be changed in the CONFIGURATION block at the top of `train_models.py`, or with `--numeric-cols` and `--categorical-cols`.

| Column | Type | Notes |
|---|---|---|
| `parcel_id` | text | Unique |
| `developed` | 0 or 1 | New-construction permit 2021–2025 |
| `x`, `y` | number | Parcel centroid in meters, EPSG:32614 (UTM 14N). `build_labels.py` writes these. Longitude/latitude also works, less accurately |
| `lot_area` | number | Square meters |
| `in_floodplain` | 0 or 1 | FEMA flood zone |
| `slope` | number | Percent |
| `elevation` | number | Meters |
| `dist_to_road` | number | Meters |
| `improvement_value_ratio` | number | Improvement value ÷ total value, 2020 appraisal roll |
| `zoning_class` | text | Collapse to the base district (for example `SF-3`), or there will be hundreds of rare categories |
| `land_cover_class` | text | NLCD 2019 class |
| `emb_A00` … `emb_A63` | number | Mean embedding inside the parcel, 2020 |
| `nbr_A00` … `nbr_A63` | number | Mean embedding in the 250 m ring, 2020 |
| `emb_change`, `nbr_change` | number | 2018 → 2020 change |
| `is_tiny_parcel` | 0 or 1 | Optional. Used for a small-parcel breakdown, not as a feature |
| `clay_000` … | number | Optional. Used only with `--include-clay` |

The tabular list is a starting point. Add or remove columns to match what you actually collect. Every tabular column must describe 2020 or earlier.

## What `train_models.py` does

- **Three models, one learner.** Standardized, L2-regularized logistic regression with balanced class weights.
- **Preprocessing inside the pipeline.** Imputation, scaling, one-hot encoding and optional PCA are fitted on the training fold only.
- **Spatial block cross-validation** (default 2 km blocks, 5 folds) is the headline. Random 5-fold runs alongside, to show how much it inflates scores.
- **Metrics.** ROC-AUC, PR-AUC, and precision, recall and lift in the top 10% of parcels.
- **Paired comparison** of model 3 against model 1: per-fold differences, a paired t-test, a Wilcoxon test, and a 95% confidence interval from a block bootstrap.
- **Placebo test.** Model 3 rerun with shuffled imagery and with random noise in place of imagery.
- **Moran's I** on out-of-fold residuals, to see whether the errors cluster in space.
- **Leakage screen.** Flags any single column that nearly separates the classes by itself, and column names that suggest post-2020 data. `--strict-leakage` stops the run.
- **Interpretation.** How much each feature family moves the predictions, plus the largest tabular coefficients.

Useful flags: `--tune-C` (choose regularization by inner cross-validation), `--pca-components 16`, `--robustness` (random forest), `--imagery-breakdown` (parcel versus neighborhood versus change), `--include-clay`, `--block-size 2000`. Run `python train_models.py --help` for all of them.

### Outputs

| File | Contents |
|---|---|
| `comparison.txt` | Plain-English readout. Start here |
| `summary.csv` | One row per model and cross-validation scheme |
| `metrics_by_fold.csv` | Every metric for every fold |
| `comparisons.csv` | Paired tests and bootstrap intervals against the parcel-only model |
| `oof_predictions.csv` | Each parcel's held-out score from each model |
| `coefficients.csv` | Model 3's coefficients (fitted on all rows, for interpretation only) |
| `auc_by_model.png`, `risk_map.png` | Figures for slides |
| `run_config.json` | The exact settings used |

Every table has a `data` column that says `SYNTHETIC` or `real`. The figures and `comparison.txt` are stamped when the data is synthetic.

## Synthetic test results (not real results)

These numbers come from fake data with a planted answer. They show the code behaves correctly. They say nothing about Austin. 9,000 fake parcels, 12.2% developed, 178 spatial blocks, 5 folds.

**Imagery signal planted** (`--imagery-signal 1`), ROC-AUC under spatial cross-validation:

| Model | ROC-AUC | PR-AUC | Top-10% lift |
|---|---|---|---|
| parcel_zoning_only | 0.724 ± 0.021 | 0.271 | 2.59× |
| imagery_only | 0.689 ± 0.028 | 0.238 | 2.39× |
| parcel_plus_imagery | 0.791 ± 0.029 | 0.380 | 3.40× |
| placebo, shuffled imagery | 0.706 | | |
| placebo, random noise | 0.700 | | |

Model 3 beat model 1 by +0.068 AUC in 5 of 5 folds (block-bootstrap 95% CI +0.048 to +0.084). Both placebos scored below model 1, as they should.

**No imagery signal** (`--imagery-signal 0`):

| Model | Spatial CV | Random CV |
|---|---|---|
| parcel_zoning_only | 0.724 | 0.722 |
| imagery_only | 0.505 | 0.581 |
| parcel_plus_imagery | 0.671 | 0.718 |

- Model 3 did not beat model 1. It was worse by 0.052, because 130 uninformative columns let the model memorize neighborhoods. With `--pca-components 16` the gap closed to 0.721 against 0.724.
- Random cross-validation gave `imagery_only` an AUC of 0.581 when the imagery held no signal. Spatial cross-validation gave 0.505, which is a coin flip. This is why the spatial number is the one to report.
- The parcel-only model scored the same under both schemes. The inflation showed up in the imagery models, which have enough columns to recognize a neighborhood.

**Leakage test.** A column that was a noisy copy of the label, and a fake `year_built` column, were both flagged by the screen. With them included, the parcel-only AUC jumped to 1.000.

## Reading your real results

- **Report the spatial cross-validation numbers.** Show the random ones only to explain the difference.
- **A null or negative result is a valid finding.** If the confidence interval for model 3 minus model 1 includes zero, say imagery did not add measurable value on top of parcel data.
- **If model 3 is worse than model 1**, try `--pca-components 16` and `--tune-C` before concluding. Report what you tried.
- **If a placebo beats model 1**, stop and look for leakage.
- **Scores are rankings, not probabilities.** Balanced class weights inflate the raw probabilities.
- **Five folds is a small sample.** A Wilcoxon test on 5 folds cannot go below p = 0.0625. Lean on the bootstrap interval and on how many folds agree.

## Limitations

- **Not yet run on real data.** Every real-data step may need small fixes on first contact.
- **Earth Engine script not run.** It was written without Earth Engine access. Run the 50-parcel pilot and inspect the CSV before the full run.
- **`build_labels.py` location mode not run.** Check the printed match counts. The permit coordinate columns are assumed to be `latitude` and `longitude`.
- **XGBoost and LightGBM paths not run.** Neither library was installed where the code was tested. The random forest path was run.
- **TCAD parcel ID field unconfirmed.** The layer's metadata lists `PROP_ID` (integer) and `geo_id` (text). The permits' `tcad_id` is 10-digit text, so `geo_id` is the likely match. Confirm it against real values from the test parcel pull.
- **2020 parcel snapshot.** The design needs parcels and appraisal values as of 2020. If only current data is available, say so. Current data can leak the outcome (a 2023 building raises today's improvement value).
- **Zoning date.** Zoning is probably available only as it is today. A rezoning that happened because of a planned project would leak.
- **A permit is a proxy.** It measures development likelihood, shaped by markets and owners, not pure physical suitability. A permit also does not mean the building was finished.
- **10 m pixels are coarse for small lots.** Parcels under about 300 m² use a centroid sample that includes the neighbors. The readout breaks results out for these.
- **Embedding dimensions are not interpretable one by one.** Interpret by feature family and by the map.
- **One city, one period.** Results describe Austin, 2021–2025.

## Attribution and AI use

The embeddings are licensed CC-BY 4.0 and require attribution: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."

AI use must be cited for this course. See `AI_ACKNOWLEDGMENT.md`.
