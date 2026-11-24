# Final presentation: outline and fill-in checklist

Drafted 2026-11-16 with Claude (Anthropic). The deck itself is a Claude Slides artifact in Gunner's account: https://claude.ai/artifact/XxeocGbXPggo3BbKfJ7S1y (private until he shares it; it can be downloaded as PowerPoint or PDF).

**The deck is a draft. It contains no real results.** Everything in orange square brackets is a placeholder. The only numbers on it are design facts (2020 cutoff, 64 embedding bands, 250 m ring, 2 km blocks) and two figures on slide 7 that come from the synthetic code test and are labeled that way.

Target: 10 minutes of talk, 5 minutes of questions. 15 slides.

## Slides

| # | Slide | Time | Placeholders to fill | Where the value comes from |
|---|---|---|---|---|
| 1 | Title | 0:20 | Presentation date | Canvas |
| 2 | The question | 0:45 | None | |
| 3 | Design: one cutoff, three models | 1:00 | Vacant/underused threshold; parcel count; developed count and share | `build_features.py` printout; top of `comparison.txt` |
| 4 | Data sources | 0:45 | Parcel source and snapshot date; roll year; zoning, floodplain and road dates | Snapshot log in `DATA_SOURCES.md` |
| 5 | Pipeline | 0:45 | Share of permits matched to a parcel | `build_labels.py` printout |
| 6 | The GeoAI part: three imagery features | 1:15 | None | |
| 7 | Evaluation: spatial CV, placebo, leakage screen | 1:15 | Optional: replace the synthetic example with the real spatial-versus-random gap | `comparison.txt` section 4 |
| 8 | Results table | 1:00 | Headline sentence; ROC-AUC, PR-AUC and lift for three models; base rate | `comparison.txt` section 1 |
| 9 | Does imagery help? | 1:15 | Verdict; AUC difference; confidence interval; folds won; two placebo scores; random-CV gain; chart | `comparison.txt` sections 2–4; `auc_by_model.png` |
| 10 | Map | 1:00 | Map figure; where it is right and wrong; Moran's I; top drivers | `risk_map.png`; sections 5–6 |
| 11 | Limitations | 0:45 | Tiny-parcel result; zoning date; parcel snapshot date | Section 7; `DATA_SOURCES.md` |
| 12–14 | One contribution slide each: Gunner, Jay, Palash | 0:20 each | Everything. Each person writes their own | |
| 15 | AI acknowledgment and data credits | 0:20 | Edit the AI sentence to match what the team did | `AI_ACKNOWLEDGMENT.md` |

Speaker notes on every slide say what to stress and repeat where each number comes from.

## Before presenting

- [ ] Search the deck for `[` and confirm no bracket is left.
- [ ] Every number on slides 3, 5, 8, 9, 10 and 11 was copied from a file in `results/` or a script printout, by a person, and a second person checked it.
- [ ] The headline on slide 8 and the verdict on slide 9 follow `RESULTS_GUIDE.md` for the outcome you actually got.
- [ ] Slide 7's synthetic example is still labeled as a code test, or has been replaced with the real gap.
- [ ] Slide 4 has no row for a layer that was not used.
- [ ] The AlphaEarth attribution sentence is still on slide 15 (its license requires it).
- [ ] One timed run-through at 10 minutes or under.

## Questions to prepare for

1. Why AlphaEarth embeddings and not Clay or Prithvi? (Clay needs a fixed square chip; per-pixel embeddings can be averaged over the real lot shape. This followed the proposal feedback.)
2. How do you avoid spatial leakage? (Whole 2 km blocks are held out; the placebo test checks the gain is not an artifact.)
3. What counts as "developed"? (A residential or commercial permit whose work class contains "New", issued 2021–2025. State the exact filter you ended up using.)
4. Why logistic regression? (One simple learner for all three models, so only the features differ. Random forest is run as a robustness check.)
5. Could the features see the future? (Every feature is dated 2020 or earlier, with the exceptions listed on the limitations slide.)
