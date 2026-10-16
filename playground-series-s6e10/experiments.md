# Experiment log - S6E10 Predicting Airline Satisfaction

Competition: https://www.kaggle.com/competitions/playground-series-s6e10 (metric: ROC AUC; public LB uses ~20% of test data)

Rules of the log:
- Every experiment gets a row, including failures. A failed experiment is still a result.
- All validation AUCs use the same stratified 80/20 split, seed 42, unless the "validation" column says otherwise - so numbers in this table are comparable.
- LB AUC is recorded only when actually observed after a real submission. Never estimate it, and never compare a local val AUC against leaderboard ranks - the public LB scores a different sample (~20% of test), so local-to-LB comparisons are only loose expectations, not rank predictions.
- One change per experiment, so the log shows what actually moves the score.
- Exp 2/3 use the holdout split both for early stopping and for evaluation, so those val AUCs are mildly optimistic (selection on the evaluated set). Fine for direction-finding; before trusting a tuned config, re-score with k-fold OOF and log it separately.

Leaderboard reference (checked 2026-10-16): #1 = 0.96273; ranks ~19-50 compressed at 0.96190; ~2,070 teams.

| id | date | model / change | config | seed | validation | val AUC | LB AUC | runtime | verdict / notes |
|----|------|----------------|--------|------|------------|---------|--------|---------|-----------------|
| 0 | 2026-10-16 | logistic regression baseline (hello.ipynb re-scored) | median impute + scaler + one-hot, max_iter=200 | 42 | stratified 80/20 | 0.92046 | 0.92227 | ~1 min | baseline; LB from Andrea's own submission (23:11 BST). LB - local = +0.0018 |
| 1 | 2026-10-16 | random forest (random_forest.ipynb) | 200 trees, defaults, no scaler | 42 | stratified 80/20 | 0.95351 | 0.95367 | 67s | keep; +0.03305 vs Exp 0. LB from Andrea's own submission (23:44 BST). LB - local = +0.0002: the holdout tracks the public LB well |
| 2 | 2026-10-16 | LightGBM raw features (lightgbm.ipynb) | lr=0.05, num_leaves=63, early stop 50 | 42 | stratified 80/20 (holdout also used for early stop) | 0.95760 | | 23s (whole notebook) | keep; +0.00409 vs Exp 1. best_iteration=418; winner -> submission_lgbm.csv |
| 3 | 2026-10-16 | LightGBM + engineered (total_delay, delay_diff, service_mean) | same as Exp 2, one change | 42 | stratified 80/20 (holdout also used for early stop) | 0.95748 | | 23s (whole notebook) | no gain on this split (-0.00012 vs Exp 2, within single-split noise - NOT evidence the features hurt). Set aside; revisit under k-fold. best_iteration=290 |

## Run notes

- 2026-10-16: first execution of Exp 3 failed with a LightGBMError - the authored scoring line mistakenly evaluated the engineered model on the NON-engineered validation frame (predict_proba(X_val) instead of X_val_e). Fixed on review and the whole notebook rerun cleanly; the numbers above are from the corrected run. Logged because silent reruns hide exactly this kind of bug.

## Next candidates (pick one per session, in rough order)

- k-fold (5-fold stratified) OOF re-score of the best model: required before any tuning decisions, since Exp 2/3 selected tree count on the evaluated holdout
- num_leaves / learning_rate variation on the winner, one at a time
- handle the 292 missing Arrival Delay values deliberately vs LightGBM native NaN
- blend RF + LightGBM predictions (mean of probabilities)
- original passenger-survey dataset as extra training data (legit external data; check licensing/provenance)
