import json
from pathlib import Path

import pytest

from student_success import TARGET_COLUMN
from student_success.data import load_dataset
from student_success.features import get_feature_columns
from student_success.preprocessing import build_preprocessor
from student_success.split import load_split_indices

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def test_dataset_matches_expected_contract():
    frame = load_dataset()
    assert frame.shape == (4424, 37)
    assert set(frame[TARGET_COLUMN]) == {"Dropout", "Enrolled", "Graduate"}


def test_scenarios_exclude_target_and_second_semester():
    enrollment = get_feature_columns("enrollment_only")
    first_semester = get_feature_columns("first_semester")
    for columns in (enrollment, first_semester):
        assert TARGET_COLUMN not in columns
        assert not any("Curricular units 2nd sem" in column for column in columns)
    assert not any("Curricular units 1st sem" in column for column in enrollment)
    assert any("Curricular units 1st sem" in column for column in first_semester)


def test_split_is_complete_disjoint_and_stratified():
    frame = load_dataset()
    train_index, test_index = load_split_indices(PROCESSED / "split_indices.npz")
    assert len(train_index) == 3539
    assert len(test_index) == 885
    assert set(train_index).isdisjoint(test_index)
    assert set(train_index).union(test_index) == set(range(len(frame)))


def test_preprocessors_are_unfitted_and_accept_both_scenarios():
    frame = load_dataset()
    for scenario in ("enrollment_only", "first_semester"):
        X = frame[get_feature_columns(scenario)]
        for model_kind in ("logistic_regression", "random_forest"):
            preprocessor = build_preprocessor(X, model_kind)
            assert not hasattr(preprocessor, "transformers_")


def test_preprocessor_fits_on_training_rows_only():
    frame = load_dataset()
    train_index, _ = load_split_indices(PROCESSED / "split_indices.npz")
    X_train = frame.loc[train_index, get_feature_columns("first_semester")]

    transformed = build_preprocessor(X_train, "random_forest").fit_transform(X_train)

    assert transformed.shape[0] == len(train_index)


def test_feature_metadata_matches_scenarios():
    metadata = json.loads((PROCESSED / "feature_metadata.json").read_text())
    for scenario in ("enrollment_only", "first_semester"):
        assert metadata[scenario]["feature_columns"] == get_feature_columns(scenario)


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_feature_columns("second_semester")
