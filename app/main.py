from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data" / "processed" / "feature_metadata.json"
DATA_PATH = ROOT / "data" / "processed" / "student_success.csv"
# Agreed handoff location for the fitted first-semester pipeline exported by Person B.
MODEL_PATH = ROOT / "models" / "first_semester_pipeline.joblib"


@st.cache_resource
def load_pipeline(path: Path) -> Any:
    return joblib.load(path)


def predict(model: Any, input_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    label = str(model.predict(input_df)[0])
    probabilities = pd.DataFrame(
        {"Outcome": model.classes_, "Probability": model.predict_proba(input_df)[0]}
    ).sort_values("Probability", ascending=False, ignore_index=True)
    return label, probabilities


@st.cache_data
def load_feature_metadata() -> dict[str, Any]:
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_reference_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def default_profiles() -> dict[str, dict[str, float | int]]:
    return {
        "At-risk example": {
            "Marital status": 1,
            "Application mode": 17,
            "Application order": 5,
            "Course": 171,
            "Daytime/evening attendance": 1,
            "Previous qualification": 1,
            "Previous qualification (grade)": 122.0,
            "Nacionality": 1,
            "Mother's qualification": 19,
            "Father's qualification": 12,
            "Mother's occupation": 5,
            "Father's occupation": 9,
            "Admission grade": 127.3,
            "Displaced": 1,
            "Educational special needs": 0,
            "Gender": 1,
            "Age at enrollment": 20,
            "International": 0,
            "Curricular units 1st sem (credited)": 0,
            "Curricular units 1st sem (enrolled)": 0,
            "Curricular units 1st sem (evaluations)": 0,
            "Curricular units 1st sem (approved)": 0,
            "Curricular units 1st sem (grade)": 0.0,
            "Curricular units 1st sem (without evaluations)": 0,
        },
        "Stable example": {
            "Marital status": 1,
            "Application mode": 1,
            "Application order": 1,
            "Course": 9500,
            "Daytime/evening attendance": 1,
            "Previous qualification": 1,
            "Previous qualification (grade)": 142.0,
            "Nacionality": 1,
            "Mother's qualification": 19,
            "Father's qualification": 38,
            "Mother's occupation": 7,
            "Father's occupation": 10,
            "Admission grade": 128.4,
            "Displaced": 1,
            "Educational special needs": 0,
            "Gender": 0,
            "Age at enrollment": 18,
            "International": 0,
            "Curricular units 1st sem (credited)": 0,
            "Curricular units 1st sem (enrolled)": 7,
            "Curricular units 1st sem (evaluations)": 9,
            "Curricular units 1st sem (approved)": 7,
            "Curricular units 1st sem (grade)": 13.3,
            "Curricular units 1st sem (without evaluations)": 0,
        },
        "Uncertain example": {
            "Marital status": 1,
            "Application mode": 18,
            "Application order": 1,
            "Course": 9238,
            "Daytime/evening attendance": 1,
            "Previous qualification": 1,
            "Previous qualification (grade)": 137.0,
            "Nacionality": 1,
            "Mother's qualification": 19,
            "Father's qualification": 38,
            "Mother's occupation": 5,
            "Father's occupation": 8,
            "Admission grade": 137.4,
            "Displaced": 1,
            "Educational special needs": 0,
            "Gender": 0,
            "Age at enrollment": 18,
            "International": 0,
            "Curricular units 1st sem (credited)": 0,
            "Curricular units 1st sem (enrolled)": 6,
            "Curricular units 1st sem (evaluations)": 10,
            "Curricular units 1st sem (approved)": 1,
            "Curricular units 1st sem (grade)": 12.0,
            "Curricular units 1st sem (without evaluations)": 0,
        },
    }


def main() -> None:
    st.set_page_config(page_title="Student Success Predictor | DHBW", layout="wide")

    metadata = load_feature_metadata()["first_semester"]
    df_reference = load_reference_dataset()
    feature_columns: list[str] = metadata["feature_columns"]
    categorical_columns = set(metadata["categorical_columns"])

    st.title("Student Success Predictor")
    st.caption(
        "Choose an example student, edit the values, and predict whether the student "
        "graduates, drops out, or is still enrolled."
    )

    model_ready = MODEL_PATH.exists()
    if not model_ready:
        st.info(
            "Waiting for the trained model. Predictions start working once "
            f"`{MODEL_PATH.relative_to(ROOT)}` is added by the modeling owner."
        )

    profiles = default_profiles()
    profile_name = st.selectbox("Example profile", list(profiles))
    profile_values = profiles[profile_name]

    st.subheader("Student inputs")
    edited_values: dict[str, float | int] = {}
    with st.form("student_profile_form"):
        col_left, col_right = st.columns(2)
        for idx, col in enumerate(feature_columns):
            with col_left if idx % 2 == 0 else col_right:
                # Profile name in the key resets the fields when another profile is chosen.
                key = f"{profile_name}_{col}"
                default_value = profile_values[col]
                if col in categorical_columns:
                    choices = sorted(df_reference[col].unique().tolist())
                    edited_values[col] = st.selectbox(
                        col, choices, index=choices.index(default_value), key=key
                    )
                elif pd.api.types.is_integer_dtype(df_reference[col]):
                    edited_values[col] = st.number_input(
                        col, min_value=0, value=int(default_value), step=1, key=key
                    )
                else:
                    edited_values[col] = st.number_input(
                        col, min_value=0.0, value=float(default_value), step=0.1, key=key
                    )

        submitted = st.form_submit_button(
            "Predict" if model_ready else "Preview input", type="primary"
        )

    if not submitted:
        return

    input_df = pd.DataFrame([edited_values], columns=feature_columns)
    st.subheader("Model input")
    st.dataframe(input_df, hide_index=True, use_container_width=True)

    if not model_ready:
        return

    label, probabilities = predict(load_pipeline(MODEL_PATH), input_df)
    st.subheader("Prediction")
    st.metric("Predicted outcome", label)
    st.dataframe(
        probabilities.style.format({"Probability": "{:.1%}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "The outcome is the recorded status at the end of the normal program duration, "
        "not a predicted dropout date."
    )


if __name__ == "__main__":
    main()
