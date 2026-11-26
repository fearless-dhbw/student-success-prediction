# Person A handoff — Logistic Regression baseline

## Reproduce

```bash
PYTHONPATH=src python src/train_logistic_regression.py
```

The command loads `data/processed/student_success.csv`, reuses
`data/processed/split_indices.npz`, runs five-fold stratified CV only on the
3,539 training rows, then evaluates the fixed configuration once on the 885
held-out rows.

## Shared outputs

- `results/baseline_metrics.csv`: tidy train, CV, and test metrics for both scenarios.
- `results/logistic_regression_classification_reports.csv`: per-class reports.
- `results/logistic_regression_confusion_matrices.csv`: fixed-order confusion-matrix counts.
- `artifacts/logistic_regression_{scenario}.joblib`: full fitted Pipeline.
- `artifacts/logistic_regression_{scenario}_metadata.json`: scenario, raw feature order, labels, and estimator settings.

## Contract for Person B

Use `student_success.evaluation.evaluate_classifier` and its metric fields for
Random Forest results. Reuse `get_feature_columns(scenario)` and the persisted
split; do not replace either. Append Random Forest rows to a separate output
while tuning, then use the same columns for the final comparison table. The
Logistic Regression artifacts are baselines, not the selected demo model.

## Caveats

The first-semester scenario may use only its six approved first-semester
features in addition to enrollment inputs. Both scenarios exclude second-semester,
target, and time-ambiguous fields. Scores describe retrospective recorded
outcomes, not dropout timing or causal risk.
