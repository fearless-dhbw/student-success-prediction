"""Create the validated shared dataset, feature metadata, and frozen split."""

import json
from pathlib import Path

import pandas as pd

from student_success.data import PROJECT_ROOT, validate_dataset
from student_success.features import SCENARIOS, get_column_types
from student_success.split import save_split_indices

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "uci_697_source" / "data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    frame = pd.read_csv(RAW_PATH, sep=";")
    frame.columns = [column.replace("\ufeff", "").strip() for column in frame.columns]
    validate_dataset(frame)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_DIR / "student_success.csv", index=False)
    save_split_indices(
        frame["Target"],
        PROCESSED_DIR / "split_indices.npz",
        PROCESSED_DIR / "split_metadata.json",
    )

    feature_metadata = {}
    for scenario, columns in SCENARIOS.items():
        categorical, numeric = get_column_types(scenario)
        feature_metadata[scenario] = {
            "feature_columns": columns,
            "categorical_columns": categorical,
            "numeric_columns": numeric,
        }
    (PROCESSED_DIR / "feature_metadata.json").write_text(
        json.dumps(feature_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated {frame.shape[0]} rows and {frame.shape[1]} columns.")


if __name__ == "__main__":
    main()
