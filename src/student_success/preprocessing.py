"""Train-only preprocessing shared by future models."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, assert_safe_feature_columns


def build_preprocessor(X_train, model_kind: str) -> ColumnTransformer:
    """Build an unfitted preprocessor for the input columns in X_train.

    The returned object must be fitted inside a model Pipeline on training data only.
    """
    if model_kind not in {"logistic_regression", "random_forest"}:
        raise ValueError("model_kind must be 'logistic_regression' or 'random_forest'.")

    assert_safe_feature_columns(list(X_train.columns))
    categorical = [column for column in X_train.columns if column in CATEGORICAL_COLUMNS]
    numeric = [column for column in X_train.columns if column in NUMERIC_COLUMNS]
    if len(categorical) + len(numeric) != len(X_train.columns):
        unknown = set(X_train.columns).difference(categorical, numeric)
        raise ValueError(f"Columns lack a declared type: {sorted(unknown)}")

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_kind == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )
