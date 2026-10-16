# Experiment log - S6E10 Predicting Airline Satisfaction

Competition: https://www.kaggle.com/competitions/playground-series-s6e10 (metric: ROC AUC; public LB uses ~20% of test data)

Rules of the log:
- Every experiment gets a row, including failures. A failed experiment is still a result.
- All validation AUCs use the same stratified 80/20 split, seed 42, unless the "validation" column says otherwise - so numbers in this table are comparable.
- LB AUC is recorded only when actually observed after a real submission. Never estimate it, and never compare a local val AUC against leaderboard ranks - the public LB scores a different sample (~20% of test), so local-to-LB comparisons are only loose expectations, not rank predictions.
- One change per experiment, so the log shows what actually moves the score.
- Exp 2/3/6c use the holdout split both for early stopping and for evaluation, so those val AUCs are mildly optimistic (selection on the evaluated set). Fine for direction-finding; before trusting a tuned config, re-score with k-fold OOF and log it separately.
- A delta on a single split is a hint, not a verdict. Small deltas (positive or negative) are logged as "unconfirmed" until a k-fold OOF re-score confirms or kills them.
- OOF numbers are a sturdier but still descriptive estimate: folds share training rows, the folds overlap the old holdout, and this dataset has been selected on repeatedly. OOF deltas are logged as "working signal" / "no signal", never as confirmed effects.

Leaderboard reference (checked 2026-10-16): #1 = 0.96273; ranks ~19-50 compressed at 0.96190; ~2,070 teams.

| id | date | model / change | config | seed | validation | val AUC | LB AUC | runtime | verdict / notes |
|----|------|----------------|--------|------|------------|---------|--------|---------|-----------------|
| 0 | 2026-10-16 | logistic regression baseline (hello.ipynb re-scored) | median impute + scaler + one-hot, max_iter=200 | 42 | stratified 80/20 | 0.92046 | 0.92227 | ~1 min | baseline; LB from Andrea's own submission (23:11 BST). LB - local = +0.0018 |
| 1 | 2026-10-16 | random forest (random_forest.ipynb) | 200 trees, defaults, no scaler | 42 | stratified 80/20 | 0.95351 | 0.95367 | 67s | keep; +0.03305 vs Exp 0. LB from Andrea's own submission (23:44 BST). LB - local = +0.0002: the holdout tracks the public LB well |
| 2 | 2026-10-16 | LightGBM raw features (lightgbm.ipynb) | lr=0.05, num_leaves=63, early stop 50 | 42 | stratified 80/20 (holdout also used for early stop) | 0.95760 | | 23s (whole notebook) | keep; +0.00409 vs Exp 1. best_iteration=418; winner -> submission_lgbm.csv |
| 3 | 2026-10-16 | LightGBM + engineered (total_delay, delay_diff, service_mean) | same as Exp 2, one change | 42 | stratified 80/20 (holdout also used for early stop) | 0.95748 | | 23s (whole notebook) | -0.00012 vs Exp 2 on this split - too small to call; unconfirmed until the fold check. Set aside; revisit under k-fold. best_iteration=290 |
| 4 | 2026-10-17 | RF tuned: 400 trees, min_samples_leaf=5 | n_est=400, leaf=5, rest=Exp 1 | 42 | stratified 80/20 | 0.95551 | | 234s (whole round-2 notebook) | new RF reference: +0.00200 vs Exp 1 on this split; still below LightGBM |
| 5 | 2026-10-17 | RF + max_features=0.5 | as Exp 4, one knob | 42 | stratified 80/20 | 0.95513 | | (same run) | -0.00038 vs Exp 4 on this split - too small to call; unconfirmed until folds |
| 6a | 2026-10-17 | logistic + engineered set (10 features) | Exp 0 pipeline, 31 features | 42 | stratified 80/20 | 0.92796 | | (same run) | +0.00750 vs Exp 0 on this split - largest round-2 delta, unconfirmed until folds; still far behind trees |
| 6b | 2026-10-17 | Exp 4 RF config + engineered set | as Exp 4, 31 features | 42 | stratified 80/20 | 0.95534 | | (same run) | -0.00016 vs Exp 4 on this split - too small to call; unconfirmed until folds |
| 6c | 2026-10-17 | LightGBM + engineered set | as Exp 2, 31 features | 42 | stratified 80/20 (early stop on holdout) | 0.95740 | | (same run) | -0.00020 vs Exp 2 on this split - too small to call; unconfirmed until folds. best_iteration=363 |
| 7 | 2026-10-17 | blend 50/50: Exp 4 RF + Exp 2 LightGBM | mean of probabilities | 42 | stratified 80/20 | 0.95690 | | (same run) | -0.00070 vs best single (Exp 2) on this split - no evidence blending helps here; re-tested on OOF in round 3 |
| 8 | 2026-10-17 | OOF checkpoint: Exp 2 raw LightGBM re-scored | 5-fold stratified OOF, inner-10% early stop per fold | 42 | 5-fold OOF (alternative partitioning of same data) | 0.95847 +/- 0.00053 | | 251s (whole round-3 notebook) | incumbent reference under OOF; folds 0.95851 0.95751 0.95907 0.95841 0.95885. Descriptive estimate, not certification - folds overlap the old holdout and the data has been selected on repeatedly |
| 9 | 2026-10-17 | OOF: LightGBM + engineered set | same protocol as Exp 8 | 42 | 5-fold OOF | 0.95831 +/- 0.00056 | | (same run) | paired fold deltas vs Exp 8 all negative (-0.00026 -0.00020 -0.00011 -0.00004 -0.00020, mean -0.00016): working signal the set slightly hurts LightGBM. Dropped for trees - no further FE investment without a new hypothesis. Stays on the table for logistic |
| 10 | 2026-10-17 | OOF: RF 400/leaf5 raw | same protocol as Exp 8 | 42 | 5-fold OOF | 0.95658 +/- 0.00042 | | (same run) | below the incumbent on all 5 folds (paired deltas about -0.0019); RF stays the second family |
| 11 | 2026-10-17 | OOF blend scan (LightGBM with RF) | weights 0.3-0.7 on OOF preds | 42 | 5-fold OOF | best 0.95832 pooled (0.7 LGBM + 0.3 RF) | | (same run) | exploratory: weight chosen on the same OOF it was scored on, and this pooled AUC is not directly comparable to the mean-of-fold numbers above (pooled per-model numbers pending). No action; incumbent stands |

## Run notes

- 2026-10-16: first execution of Exp 3 failed with a LightGBMError - the authored scoring line mistakenly evaluated the engineered model on the NON-engineered validation frame (predict_proba(X_val) instead of X_val_e). Fixed on review and the whole notebook rerun cleanly; the numbers above are from the corrected run. Logged because silent reruns hide exactly this kind of bug.
- 2026-10-17 (round 2): nothing beat Exp 2 raw LightGBM (0.95760) - every round-2 delta vs the incumbent was negative on this split. The round-2 refit cell compared candidates only within the round, so it wrote submission_round2.csv from Exp 6c (0.95740), below the incumbent. Caught on review: submission_round2.csv should NOT be submitted; submission_lgbm.csv (Exp 2) remains the best submission candidate. The notebook now carries a corrected refit cell with an incumbent guard (marked, not re-executed). The 0.960 local target was NOT reached. Mandatory next step before more tuning: 5-fold OOF re-score of the top configs - this holdout has guided many selections and its estimates drift optimistic.
- 2026-10-17 (round 3): 5-fold OOF checkpoint executed cleanly (251s wall, peak RAM 3.05GB, zero errors, no execution fixes needed). Outcomes: engineered feature set dropped for tree models (all 5 paired fold deltas negative); blend scan produced no signal over the incumbent; incumbent Exp 2 raw LightGBM stands as the only submission candidate (submission_lgbm.csv). The 0.960 target was NOT reached on OOF (best 0.95847). Two disclosures: (1) a mid-round wording revision of round3.ipynb arrived malformed (markdown prose inside a code cell, summary code removed) and was correctly rejected by review before execution - the original v1 ran; (2) in the pushed round3.ipynb, verdict labels were relabeled on review (from CONFIRMED/UNCONFIRMED to WORKING SIGNAL/NO SIGNAL) in source and captured output together - numeric results untouched. Round-4 options per notebook: one-knob tuning under OOF, or the original passenger-survey dataset as extra training data - the latter needs Andrea's explicit OK (provenance/licensing) before any use.

## Next candidates (pick one per session, in rough order)

- round 4 (running): one-knob moves under the same OOF protocol - RF min_samples_leaf=10, RF max_depth=25, LightGBM learning_rate=0.03
- num_leaves variation on the OOF winner, one at a time, selected on OOF
- handle the 292 missing Arrival Delay values deliberately vs LightGBM native NaN
- original passenger-survey dataset as extra training data (legit external data; check licensing/provenance, needs Andrea's OK first)
