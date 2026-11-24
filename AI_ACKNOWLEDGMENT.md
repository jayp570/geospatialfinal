# AI use acknowledgment

The code in this folder was written with the help of Claude, an AI assistant made by Anthropic, working from the team's project design.

**What Claude wrote or drafted**

- `train_models.py`, `make_synthetic_data.py` and `ee_extract_embeddings.py`
- The location-matching and cutoff options in `build_labels.py`, and the `--since` option in `pull_permits.py`
- The original versions of `pull_permits.py`, `pull_parcels.py` and `build_labels.py`
- `README.md`
- Added on 2026-11-16, each in its own pull request: `build_features.py`, the terrain and land-cover option in `ee_extract_embeddings.py`, `pipeline_colab.ipynb`, the tests in `tests/`, the fix to the service address in `pull_parcels.py`, and the documents in `docs/` (data sources, results guide, presentation outline)
- A first draft of the final slide deck, with placeholders in place of every result

**What Claude did not do**

- Claude did not download or see any real Austin parcel, permit or imagery data.
- Claude ran the training code only on synthetic data made by `make_synthetic_data.py`. Those results test the code and are not findings.
- The Earth Engine script was not run against Earth Engine by Claude. Its tests use a stand-in for the Earth Engine library.
- The location-matching code was run by Claude only on small fake parcels and permits in `tests/`.
- Claude did not write any result, number or conclusion about Austin. The draft deck's result slides were left as placeholders.

**What the team is responsible for**

- Collecting the real data, running the pipeline, and checking the outputs
- Reading and understanding the code well enough to explain and defend it
- All results, interpretation and conclusions in the presentation

**Suggested wording for the slides or report**

> We used Claude (Anthropic) to help write and debug our data pipeline and model comparison code, and to help draft project documents. We ran all analyses on real data ourselves and are responsible for the results and conclusions.

Edit the wording so it matches what the team actually did.
