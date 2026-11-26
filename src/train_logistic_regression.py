"""Run Person A's reproducible Logistic Regression baseline."""

from student_success.logistic_regression import run_training

if __name__ == "__main__":
    metrics = run_training()
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
