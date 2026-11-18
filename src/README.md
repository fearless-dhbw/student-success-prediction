# Training and evaluation code

Place the data preparation, model training, and evaluation code here. Agree on shared input column names, the target labels, and the split before working on separate models.

- A implements data preparation, Logistic Regression, and the feature-subset experiment.
- B implements the shared split and metrics, tunes Random Forest, compares results, and exports the selected first-semester pipeline.
- Export held-out labels and predictions with a stable row identifier so C can reproduce the confusion matrix.
- Record the training and evaluation commands in the root README once the entry points exist.
