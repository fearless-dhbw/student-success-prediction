# Streamlit demo

One-page app in `app/main.py`. Visitors choose an example student profile, edit the inputs, and get the selected first-semester model's prediction. The DHBW colors are set in `.streamlit/config.toml`.

Run locally from the repository root, so that the theme is applied:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Model handoff (Person B → Person C)

The app loads exactly one file: `models/first_semester_pipeline.joblib`. Until that file exists, the app shows the model input but makes no prediction.

The exported file must be:

- a fitted scikit-learn `Pipeline` (shared preprocessor + classifier), saved with `joblib.dump(pipeline, "models/first_semester_pipeline.joblib")`
- trained on `get_feature_columns("first_semester")` (24 raw columns, the same order as `data/processed/feature_metadata.json`)
- trained on the string labels `Dropout`, `Enrolled`, `Graduate`, with `predict_proba` available

## Remaining steps (blocked until the model file exists)

- check that the same example gives the same prediction in the training code and in the app
- deploy publicly, check the link in a signed-out browser, and add it to the root `README.md`
