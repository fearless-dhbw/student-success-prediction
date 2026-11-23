# Sprint 1 handoff

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
```

The committed, validated shared dataset is `data/processed/student_success.csv`; no download is needed to run the tests. To regenerate it from the official source:

```bash
PYTHONPATH=src python src/download_data.py
PYTHONPATH=src python src/prepare_data.py
```

The official source remains local at ignored `data/raw/uci_697_source/data.csv`.

## Shared contract for B and C

- Target: `Target`; labels: `Dropout`, `Enrolled`, `Graduate`.
- Scenarios: call `get_feature_columns("enrollment_only")` or `get_feature_columns("first_semester")`.
- Split: `data/processed/split_indices.npz`, generated with stratified 80/20 split, `random_state=42`. Do not create another split.
- Preprocessing: `build_preprocessor(X_train, "random_forest")` or `build_preprocessor(X_train, "logistic_regression")`. It is unfitted and must be fitted inside a `Pipeline` on training folds only.
- UI metadata: `data/processed/feature_metadata.json` contains the expected raw first-semester input order and types.

Person B can now start Random Forest work using the frozen split and shared preprocessing. Person C can start the UI schema and input controls, but no fitted pipeline is available until model training in a later sprint.
