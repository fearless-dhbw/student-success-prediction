"""Explicit, time-safe feature scenarios."""

from . import TARGET_COLUMN

ENROLLMENT_ONLY_COLUMNS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Gender",
    "Age at enrollment",
    "International",
]

FIRST_SEMESTER_COLUMNS = [
    *ENROLLMENT_ONLY_COLUMNS,
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Gender",
    "International",
]

NUMERIC_COLUMNS = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

SCENARIOS = {
    "enrollment_only": ENROLLMENT_ONLY_COLUMNS,
    "first_semester": FIRST_SEMESTER_COLUMNS,
}


def get_feature_columns(scenario: str) -> list[str]:
    """Return ordered raw inputs for one approved prediction moment."""
    try:
        return SCENARIOS[scenario].copy()
    except KeyError as error:
        raise ValueError(f"Unknown scenario {scenario!r}. Choose from {list(SCENARIOS)}.") from error


def get_column_types(scenario: str) -> tuple[list[str], list[str]]:
    """Return categorical and numeric inputs for an approved scenario."""
    columns = get_feature_columns(scenario)
    categorical = [column for column in columns if column in CATEGORICAL_COLUMNS]
    numeric = [column for column in columns if column in NUMERIC_COLUMNS]
    return categorical, numeric


def assert_safe_feature_columns(feature_columns: list[str]) -> None:
    """Reject targets and future-semester inputs before model fitting."""
    forbidden = [
        column
        for column in feature_columns
        if column == TARGET_COLUMN or "Curricular units 2nd sem" in column
    ]
    if forbidden:
        raise ValueError(f"Forbidden model inputs: {forbidden}")
