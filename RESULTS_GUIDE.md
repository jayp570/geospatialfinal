# What to say about the result

Written 2026-11-16 with Claude (Anthropic), before any real result exists. It contains no results. It says how to read `results/comparison.txt` and how to word each outcome honestly.

## First, decide which case you are in

Open `comparison.txt`, section 2. Look at one line: the **95% block-bootstrap interval** for model 3 (parcel plus imagery) minus model 1 (parcel only).

| The interval is... | Case |
|---|---|
| Entirely above zero | A. Imagery helped |
| Includes zero | B. No measurable difference |
| Entirely below zero | C. Imagery made it worse |

Use the interval, not the p-values. With five folds a Wilcoxon test cannot go below 0.0625, so "not significant" there means little.

**Before trusting any case, check three things:**

1. **No leakage warning** at the top of `comparison.txt`. If there is one, fix it first.
2. **Placebos (section 3) are at or below zero.** If shuffled or random imagery "helps", something is leaking and no case applies yet.
3. **You are reading the spatial numbers,** not the random cross-validation ones.

## Case A: imagery helped

**Say:** "Adding the satellite embeddings improved how well the model ranked parcels. ROC-AUC rose from [model 1] to [model 3] under spatial cross-validation, a gain of [difference], with a 95% interval of [low] to [high]. The imagery model was ahead in [n] of 5 folds, and neither placebo showed a gain."

**Then add:**
- How big the gain is in plain terms. Use lift: "the top 10% of parcels held [x] times the base rate, up from [y]."
- Which part of the imagery carried it (section 8, extra feature sets): the parcel itself, the 250 m surroundings, or the 2018–2020 change.
- Whether random forest agrees (robustness section).

**Do not say:** that imagery "causes" or "explains" development, that a given embedding band means something, or that the scores are probabilities.

**If a placebo gained at least half as much as the real imagery,** `comparison.txt` prints a caution. Then say the gain may be partly an artifact and you could not rule that out.

## Case B: no measurable difference

**Say:** "Adding the satellite embeddings did not measurably change the ranking. The difference was [difference], and the 95% interval, [low] to [high], includes zero. For this city and period, parcel and zoning data already carried the signal we could detect."

**Then add:**
- This is a finding, not a failure. The project was designed so a null result could be trusted: same rows, same folds, same learner, and a placebo test.
- What the imagery-only model scored. If it is clearly above 0.5, the embeddings do contain development signal; it overlaps with what the tabular columns already say. If it is near 0.5, they contain little.
- What you tried: `--tune-C`, and `--pca-components 16`. Report both, including the ones that changed nothing.
- The honest limits on a null: five folds, 10 m pixels on small lots, one city.

**Do not say:** "imagery does not help predict development." You showed it did not help *this model, here, measurably*. Also do not hunt through settings until something crosses zero and present only that.

## Case C: imagery made it worse

**Say:** "Adding the satellite embeddings made the ranking worse, by [difference] (interval [low] to [high]). The likely reason is overfitting: about 130 extra columns let the model memorize neighborhoods in the training blocks, and that does not transfer to held-out blocks."

**Then do, and report, in this order:**
1. Rerun with `--tune-C` if it was off.
2. Rerun with `--pca-components 16`, which squeezes the imagery columns down to 16.
3. Compare the random cross-validation gain (section 4). If imagery "helps" under random splits and hurts under spatial ones, that gap is your evidence for the memorizing explanation, and it is the most interesting slide you have.

**Then say which of these happened:**
- The gap closed after PCA: "With the embeddings compressed, the two models were indistinguishable." You are now in case B, and you say you got there by reducing dimensions.
- The gap stayed: report it as a negative result.

**Do not say:** nothing. A negative result that is explained is a good final presentation. Do not switch the headline to the random cross-validation numbers because they look better.

## In every case

- Report **spatial** cross-validation as the headline. Show the random numbers only to explain the gap.
- Scores are **rankings**, not probabilities. Balanced class weights inflate them.
- The label is a **permit**, which measures development likelihood, not pure physical suitability.
- State the **vacant/underused threshold** and the **"developed" definition** you used.
- If results differ for tiny parcels (section 7), say so: 10 m pixels are too coarse for them.
- Every number on a slide is copied from a results file. If a number is not in a file, it does not go on a slide.

## One-line answers for Q&A

- **"Is that difference significant?"** "The block-bootstrap interval is [low] to [high]. With five folds we rely on that and on how many folds agree, not on a p-value."
- **"Why is random cross-validation higher?"** "Neighboring parcels look alike. A random split tests the model on parcels whose neighbors it trained on."
- **"What would you do next?"** Pick one you believe: a second city, a finer-resolution embedding, a 2020 zoning snapshot, or predicting completed buildings instead of permits.
