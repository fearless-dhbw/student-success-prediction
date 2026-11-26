"""Leakage-safe Logistic Regression baseline for both approved scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import CLASS_LABELS, TARGET_COLUMN
from .data import get_features_and_target, load_dataset
from .evaluation import METRIC_COLUMNS, evaluate_classifier, evaluate_cross_validation
from .features import get_feature_columns
from .preprocessing import build_preprocessor
from .split import SEED, load_split_indices

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "figures" / "logistic_regression"
SCENARIOS = ("enrollment_only", "first_semester")
MODEL_NAME = "Logistic Regression"
OWNER = "Person A"
CV_FOLDS = 5


def build_logistic_regression_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """Return an unfitted full pipeline for an approved raw feature frame."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X_train, "logistic_regression")),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _scoring() -> dict:
    return {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "dropout_precision": make_scorer(
            precision_score, labels=["Dropout"], average="macro", zero_division=0
        ),
        "dropout_recall": make_scorer(
            recall_score, labels=["Dropout"], average="macro", zero_division=0
        ),
    }


def _save_confusion_matrix(matrix: pd.DataFrame, scenario: str) -> None:
    """Save a readable test confusion-matrix chart with the fixed class order."""
    counts = matrix.pivot(index="actual", columns="predicted", values="count").reindex(
        index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(counts, cmap="Blues")
    ax.set_xticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    ax.set_yticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    ax.set_xlabel("Predicted recorded outcome")
    ax.set_ylabel("Actual recorded outcome")
    ax.set_title(f"Logistic Regression: {scenario.replace('_', ' ')}")
    threshold = counts.to_numpy().max() / 2
    for row, actual in enumerate(CLASS_LABELS):
        for column, predicted in enumerate(CLASS_LABELS):
            value = counts.loc[actual, predicted]
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.colorbar(image, ax=ax, label="Students")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"confusion_matrix_{scenario}.png", bbox_inches="tight")
    plt.close(fig)


def train_scenario(frame: pd.DataFrame, scenario: str) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """Cross-validate, fit, test, and persist one fixed-scenario baseline."""
    feature_columns = get_feature_columns(scenario)
    X, y = get_features_and_target(frame, feature_columns)
    train_index, test_index = load_split_indices(PROCESSED_DIR / "split_indices.npz")
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    pipeline = build_logistic_regression_pipeline(X_train)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    cv_scores, cv_seconds = evaluate_cross_validation(pipeline, X_train, y_train, cv, _scoring())
    cv_record = {
        "owner": OWNER,
        "scenario": scenario,
        "model": MODEL_NAME,
        "split": f"{CV_FOLDS}-fold CV (training only)",
        "n_samples": len(y_train),
        "accuracy": cv_scores["accuracy"][0],
        "macro_f1": cv_scores["macro_f1"][0],
        "weighted_f1": cv_scores["weighted_f1"][0],
        "dropout_precision": cv_scores["dropout_precision"][0],
        "dropout_recall": cv_scores["dropout_recall"][0],
        "train_seconds": cv_seconds,
        "seed": SEED,
        "cv_std_macro_f1": cv_scores["macro_f1"][1],
    }

    started = perf_counter()
    fitted_pipeline = clone(pipeline).fit(X_train, y_train)
    train_seconds = perf_counter() - started
    train_record, train_report, _ = evaluate_classifier(
        fitted_pipeline,
        X_train,
        y_train,
        "train",
        scenario,
        owner=OWNER,
        model_name=MODEL_NAME,
        train_seconds=train_seconds,
    )
    test_record, test_report, test_matrix = evaluate_classifier(
        fitted_pipeline,
        X_test,
        y_test,
        "test",
        scenario,
        owner=OWNER,
        model_name=MODEL_NAME,
        train_seconds=train_seconds,
    )

    joblib.dump(fitted_pipeline, ARTIFACTS_DIR / f"logistic_regression_{scenario}.joblib")
    metadata = {
        "model": MODEL_NAME,
        "scenario": scenario,
        "target_column": TARGET_COLUMN,
        "class_labels": list(CLASS_LABELS),
        "feature_columns": feature_columns,
        "random_state": SEED,
        "cv_folds": CV_FOLDS,
        "estimator": {
            "C": 1.0,
            "max_iter": 2000,
            "solver": "lbfgs",
        },
    }
    (ARTIFACTS_DIR / f"logistic_regression_{scenario}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _save_confusion_matrix(test_matrix, scenario)
    return [cv_record, train_record, test_record], pd.concat(
        [train_report, test_report], ignore_index=True
    ), test_matrix


def run_training() -> pd.DataFrame:
    """Run both scenarios using the frozen split and write comparison-ready outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_dataset()
    records: list[dict] = []
    reports: list[pd.DataFrame] = []
    matrices: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        scenario_records, report, matrix = train_scenario(frame, scenario)
        records.extend(scenario_records)
        reports.append(report)
        matrices.append(matrix)

    metrics = pd.DataFrame(records)
    ordered_columns = METRIC_COLUMNS + ["cv_std_macro_f1"]
    metrics = metrics.reindex(columns=ordered_columns)
    metrics.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    pd.concat(reports, ignore_index=True).to_csv(
        RESULTS_DIR / "logistic_regression_classification_reports.csv", index=False
    )
    pd.concat(matrices, ignore_index=True).to_csv(
        RESULTS_DIR / "logistic_regression_confusion_matrices.csv", index=False
    )
    return metrics
