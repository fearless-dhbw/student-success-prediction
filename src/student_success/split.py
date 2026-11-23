"""One persisted stratified hold-out split for all models."""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from . import CLASS_LABELS

SEED = 42
TEST_SIZE = 0.2


def create_split_indices(y, seed: int = SEED, test_size: float = TEST_SIZE):
    """Create the required stratified train/test row indices."""
    indices = np.arange(len(y))
    return train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)


def save_split_indices(y, output_path: str | Path, metadata_path: str | Path) -> None:
    """Persist the shared split and enough metadata to check it later."""
    train_index, test_index = create_split_indices(y)
    np.savez(output_path, train_index=train_index, test_index=test_index)
    metadata = {
        "random_state": SEED,
        "test_size": TEST_SIZE,
        "n_rows": int(len(y)),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "class_labels": list(CLASS_LABELS),
        "train_class_counts": {str(key): int(value) for key, value in y.iloc[train_index].value_counts().sort_index().items()},
        "test_class_counts": {str(key): int(value) for key, value in y.iloc[test_index].value_counts().sort_index().items()},
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_split_indices(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen shared train and test row indices."""
    with np.load(path) as split:
        return split["train_index"], split["test_index"]
