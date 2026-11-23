# Feature availability and leakage decisions

The official UCI CSV has 4,424 rows, 37 columns, no missing cells, and no duplicate rows. The target is `Target` with `Dropout` (1,421), `Enrolled` (794), and `Graduate` (2,209). Valid zeros in semester-unit counts are kept as values, not treated as missing.

## Approved inputs

`enrollment_only` has 18 inputs: demographic and application records, parents' coded education/occupation, qualification/admission grades, age at enrollment, and enrollment-status flags. Category-coded integers are treated as categorical. `Application order`, grades, and age are numeric.

`first_semester` adds six `Curricular units 1st sem` counts/grade fields. These are numeric. It includes 24 inputs in total.

The exact ordered lists and declared types are in `data/processed/feature_metadata.json` and `src/student_success/features.py`.

## Excluded inputs

| Columns | Decision | Reason |
|---|---|---|
| `Target` | Exclude from both | Prediction label. |
| All `Curricular units 2nd sem ...` columns | Exclude from both | Future academic performance. |
| All `Curricular units 1st sem ...` columns | Exclude from enrollment-only | Not available at enrollment. |
| `Debtor`, `Tuition fees up to date`, `Scholarship holder` | Exclude from both | UCI documentation does not establish they are recorded before the prediction moment; payment/status values may change after enrollment. |
| `Unemployment rate`, `Inflation rate`, `GDP` | Exclude from both | Dataset documentation does not establish the observation date/linkage for each student. |

These conservative exclusions should be revisited only if an authoritative source establishes the fields' timing. No row index is included as a feature.
