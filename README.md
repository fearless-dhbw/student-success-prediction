# Student Success Prediction

A DHBW Data Science group project. We investigate how much first-semester results improve predictions of recorded degree outcomes compared with information available at enrollment alone.

The three outcome labels are **Dropout**, **Enrolled**, and **Graduate**. They describe students’ status at the end of the normal program duration. The dataset does not provide individual withdrawal dates, so this analysis cannot establish how far in advance an adviser could have acted.

## Experiment

| Model | Enrollment features | Enrollment + first-semester features |
|---|---|---|
| Logistic Regression baseline | Complete | Complete |
| Tuned Random Forest | Planned | Planned |

Use the same stratified train/test split for all four configurations. Fit preprocessing, select features, and tune hyperparameters using training data and cross-validation. Report held-out macro-F1, dropout precision and recall, and a confusion matrix. See the [evaluation protocol](docs/evaluation.md).

## Data

[UCI Predict Students’ Dropout and Academic Success](https://doi.org/10.24432/C5MC89) by Realinho, Vieira Martins, Machado, and Baptista (2021), licensed under CC BY 4.0. See [data preparation](data/README.md) for the inputs and processed dataset handoff.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python src/download_data.py
PYTHONPATH=src python src/prepare_data.py
python -m pytest tests -q
```

`data/raw/` is local and ignored. The validated shared CSV, frozen split, and feature metadata are in `data/processed/`.

## Logistic Regression baseline

Run Person A's two fixed, leakage-safe baselines with:

```bash
PYTHONPATH=src python src/train_logistic_regression.py
```

The command performs five-fold stratified CV on training data, evaluates each
fixed scenario once on the held-out split, and writes pipelines, metrics,
classification reports, and confusion matrices. See
[`docs/HANDOFF_PERSON_A.md`](docs/HANDOFF_PERSON_A.md) for the shared
comparison interface.

## Repository structure

| Location | Purpose |
|---|---|
| [data/](data/README.md) | Data source, feature timing, and processed dataset |
| [src/](src/README.md) | Preprocessing, training, and evaluation code |
| [app/](app/README.md) | Streamlit demo and model integration |
| [docs/evaluation.md](docs/evaluation.md) | Comparison design, metrics, and results template |
| [docs/team-and-submission.md](docs/team-and-submission.md) | Ownership, rubric, and submission checklist |

## Project status

Sprint 1 provides the validated shared dataset, conservative feature scenarios, frozen split, and unfitted preprocessing. Logistic Regression, Random Forest, and the Streamlit app are not implemented yet. Add the public Streamlit URL only after verifying live predictions in a signed-out browser.
