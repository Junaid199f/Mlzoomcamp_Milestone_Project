"""
Central configuration for the fraud detection project.

Keeping all common paths and constants in one place makes it easier to reuse
them across the EDA, training, export, and serving scripts.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "transactions.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EDA_DIR = ARTIFACTS_DIR / "eda"
MODEL_DIR = ARTIFACTS_DIR / "models"
EXPORT_DIR = ARTIFACTS_DIR / "exports"
NOTEBOOK_PATH = BASE_DIR / "e-commerce-fraud-detection-gb-acc-0-99.ipynb"
EXPORTED_NOTEBOOK_SCRIPT = EXPORT_DIR / "notebook_export.py"
SAVED_MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "best_model_metrics.json"


def ensure_directories() -> None:
    """Create directories that are expected by the rest of the project."""

    for path in (ARTIFACTS_DIR, EDA_DIR, MODEL_DIR, EXPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


ensure_directories()
