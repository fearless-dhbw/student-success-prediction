# Evaluation protocol

1. Create one reproducible, stratified train/test split and reuse it across all four model/input combinations. Record the seed and split sizes.
2. Fit data-dependent preprocessing, compare the smaller feature subset, and tune Random Forest using only training folds. Use macro-F1 for validation and record the selection rationale.
3. Select the first-semester pipeline for the demo using validation results. Score all four configurations on the held-out students.
4. Report macro-F1, dropout precision and recall, and the selected model’s confusion matrix. Discuss one recurring error and compare training and validation scores to assess overfitting.
5. Compare both feature sets within each model to measure the added value of first-semester results. Compare models within each feature set to measure the impact of model choice.

## Held-out results

| Inputs | Model | Macro-F1 | Dropout precision | Dropout recall |
|---|---|---|---|---|
| Enrollment | Logistic Regression | 0.455 | 0.560 | 0.489 |
| Enrollment | Random Forest | Pending | Pending | Pending |
| Enrollment + semester one | Logistic Regression | 0.604 | 0.719 | 0.722 |
| Enrollment + semester one | Random Forest | Pending | Pending | Pending |

Document the split, preprocessing, hyperparameters, feature subset, and observed error patterns alongside these results. Interpret the scores as performance on recorded degree outcomes. Individual dropout dates are unavailable in the public dataset.
