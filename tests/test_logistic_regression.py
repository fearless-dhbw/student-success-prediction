from sklearn.dummy import DummyClassifier

from student_success.data import get_features_and_target, load_dataset
from student_success.evaluation import METRIC_COLUMNS, evaluate_classifier
from student_success.features import get_feature_columns
from student_success.logistic_regression import build_logistic_regression_pipeline
from student_success.split import load_split_indices


def test_logistic_pipeline_is_unfitted_and_uses_expected_settings():
    frame = load_dataset()
    train_index, _ = load_split_indices("data/processed/split_indices.npz")
    X_train, _ = get_features_and_target(
        frame.iloc[train_index], get_feature_columns("first_semester")
    )

    pipeline = build_logistic_regression_pipeline(X_train)

    assert list(pipeline.named_steps) == ["preprocessor", "classifier"]
    assert not hasattr(pipeline.named_steps["preprocessor"], "transformers_")
    assert pipeline.named_steps["classifier"].max_iter == 2000
    assert pipeline.named_steps["classifier"].random_state == 42


def test_evaluation_uses_fixed_three_class_report_and_metric_contract():
    frame = load_dataset()
    X, y = get_features_and_target(frame.head(30), get_feature_columns("enrollment_only"))
    model = DummyClassifier(strategy="most_frequent").fit(X, y)

    record, report, matrix = evaluate_classifier(
        model,
        X,
        y,
        "train",
        "enrollment_only",
        owner="Person A",
        model_name="Dummy",
    )

    assert set(METRIC_COLUMNS).issubset(record)
    assert record["n_samples"] == len(y)
    assert set(["Dropout", "Enrolled", "Graduate"]).issubset(set(report["label"]))
    assert matrix["count"].sum() == len(y)
    assert set(matrix["actual"]) == {"Dropout", "Enrolled", "Graduate"}
    assert set(matrix["predicted"]) == {"Dropout", "Enrolled", "Graduate"}
