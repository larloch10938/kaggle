# Copilot instructions for this repository

## Project shape

This repository is a collection of Kaggle-style data science experiments, not a packaged application. The top-level folders are competition or task-specific workspaces (for example `housing_prices`, `titanic`, and `playground-series-s5e8`). Each folder usually contains:

- raw or downloaded CSV data (`train.csv`, `test.csv`, `sample_submission.csv`)
- one or more notebooks for exploration, EDA, and model development
- a notebook or script that prepares a final submission file

The repository root is intentionally lightweight: there is no `src/` package, no app entrypoint, and no formal library layout. The important unit of work is the notebook or folder-level experiment.

## Build, test, and lint commands

There is no repo-level build pipeline, package manifest, or lint configuration in this repository.

Expected workflow:

- Use a Python environment with Jupyter and the common ML stack available in Kaggle/Anaconda-style setups (`pandas`, `numpy`, `scikit-learn`, `xgboost`, etc.).
- Run a notebook directly in Jupyter or via a command like:

  ```bash
  jupyter nbconvert --execute --to notebook --inplace path/to/notebook.ipynb
  ```

  or

  ```bash
  python -m jupyter nbconvert --execute --to notebook --inplace path/to/notebook.ipynb
  ```

- If you need to run a single notebook as a focused validation step, execute just that notebook instead of the whole repo.

There are no project-level unit tests or lint commands configured here, so do not assume `pytest`, `ruff`, `black`, `npm`, or `make` targets exist unless you are adding them for a specific experiment.

## High-level architecture

This repo follows a notebook-first pattern:

1. A competition folder contains the dataset and its metadata.
2. Notebooks are used for EDA, preprocessing, model experimentation, and final submission generation.
3. The model logic often lives directly in notebook cells, with data cleaning, feature encoding, scaling, and hyperparameter search all in the same workbook.
4. Final outputs are usually CSV files produced for Kaggle submission rather than application artifacts.

The repeated pattern in this repo is:

- load `train.csv` / `test.csv`
- create a feature set for modeling
- encode or transform categorical variables
- train a model (for example `XGBClassifier`, logistic regression, or other scikit-learn estimators)
- validate with a metric such as ROC AUC or accuracy
- save a `submission.csv`-style output

Many notebooks are self-contained and depend on the local data files in the same directory. There is no cross-project code sharing layer yet.

## Key conventions in this repo

- Keep each competition experiment isolated in its own folder.
- Prefer working with the local data files already present in that folder rather than introducing a shared pipeline abstraction.
- Treat the training and test CSVs as the canonical data source for each experiment; notebook code commonly reads them directly from the folder.
- Data files are intentionally not tracked in git (`.gitignore` contains `*.csv`), so local datasets are expected to be downloaded or created outside the repo.
- Notebook names are descriptive and task-oriented (`data_exploration.ipynb`, `eda.ipynb`, `main_xgb.ipynb`), and they usually map to a single workflow stage.
- Common modeling patterns in this repo include feature scaling, categorical encoding with `LabelEncoder` or similar preprocessing, and hyperparameter tuning via `GridSearchCV`.
- If adding new work, follow the same competition-folder layout instead of creating a new top-level package structure unless the repo clearly evolves beyond notebook experiments.

## Practical guidance for Copilot

- When asked to work in this repo, assume the task is notebook-driven and dataset-driven rather than service-oriented.
- Prefer edits inside the relevant competition folder instead of introducing shared modules or new framework conventions.
- If a request requires a reproducible artifact, favor a notebook that mirrors the existing style: load data, preprocess, train model, and emit a submission file.
- Before proposing large-scale refactors, check whether the repo is intentionally minimal and notebook-scoped; the current structure suggests experimentation over infrastructure.
