"""Dataset loading and validation."""

from pathlib import Path

import pandas as pd

from . import CLASS_LABELS, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student_success.csv"
EXPECTED_ROWS = 4424
EXPECTED_COLUMNS = 37


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise ValueError when the shared dataset does not match the UCI contract."""
    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows and {EXPECTED_COLUMNS} columns; got {frame.shape}."
        )
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    labels = set(frame[TARGET_COLUMN].unique())
    if labels != set(CLASS_LABELS):
        raise ValueError(f"Unexpected target labels: {sorted(labels)}")


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the checked dataset used by every model and scenario."""
    frame = pd.read_csv(path)
    validate_dataset(frame)
    return frame


def get_features_and_target(frame: pd.DataFrame, feature_columns: list[str]):
    """Return a feature frame in the requested order and the target series."""
    missing = set(feature_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing requested columns: {sorted(missing)}")
    return frame.loc[:, feature_columns].copy(), frame[TARGET_COLUMN].copy()
