# Data preparation

Source: [UCI Predict Students’ Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), CC BY 4.0. Credit Realinho, Vieira Martins, Machado, and Baptista (2021), DOI 10.24432/C5MC89. Record any transformations here.

## Feature timing

- **Enrollment:** Document the exact eligible columns, such as previous qualification and grade, application route, program, and daytime/evening study. Verify the timing of administrative fields before using them.
- **After semester one:** Add first-semester measures such as modules passed and average grade. Exclude semester-two features from both configurations.
- **Target:** Keep the recorded outcome separate from inputs.

Keep the source download under `data/raw/`. Add the preparation script under `src/` and commit its cleaned CSV under `data/processed/`. Fit learned transformations inside the training pipeline after splitting. Record the column definitions and the command that regenerates the cleaned CSV here.
