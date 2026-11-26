"""Shared multiclass evaluation helpers for the project models."""

from __future__ import annotations

from time import perf_counter

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from . import CLASS_LABELS
from .split import SEED


METRIC_COLUMNS = [
    "owner",
    "scenario",
    "model",
    "split",
    "n_samples",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "dropout_precision",
    "dropout_recall",
    "train_seconds",
    "seed",
]


def evaluate_classifier(
    model,
    X,
    y,
    split_name: str,
    scenario: str,
    *,
    owner: str,
    model_name: str,
    train_seconds: float | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate a fitted classifier with the team metric and label contract."""
    predictions = model.predict(X)
    report = classification_report(
        y,
        predictions,
        labels=list(CLASS_LABELS),
        output_dict=True,
        zero_division=0,
    )
    record = {
        "owner": owner,
        "scenario": scenario,
        "model": model_name,
        "split": split_name,
        "n_samples": len(y),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "dropout_precision": report["Dropout"]["precision"],
        "dropout_recall": report["Dropout"]["recall"],
        "train_seconds": train_seconds,
        "seed": SEED,
    }
    report_frame = (
        pd.DataFrame(report).T.rename_axis("label").reset_index().assign(
            owner=owner, scenario=scenario, model=model_name, split=split_name
        )
    )
    matrix = confusion_matrix(y, predictions, labels=list(CLASS_LABELS))
    matrix_frame = pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS)
    matrix_frame.index.name = "actual"
    matrix_frame.columns.name = "predicted"
    matrix_frame = matrix_frame.reset_index().melt(
        id_vars="actual", var_name="predicted", value_name="count"
    )
    matrix_frame = matrix_frame.assign(
        owner=owner, scenario=scenario, model=model_name, split=split_name
    )
    return record, report_frame, matrix_frame


def evaluate_cross_validation(estimator, X, y, cv, scoring: dict) -> tuple[dict, float]:
    """Run training-only CV and return mean and standard deviation per metric."""
    from sklearn.model_selection import cross_validate

    started = perf_counter()
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )
    elapsed = perf_counter() - started
    summary = {
        metric: (float(scores[f"test_{metric}"].mean()), float(scores[f"test_{metric}"].std()))
        for metric in scoring
    }
    return summary, elapsed
