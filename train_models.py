"""Model training and tuning pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import METRICS_PATH, SAVED_MODEL_PATH
from data_pipeline import build_preprocessor, engineer_features, load_dataset, train_test_split_features


def _model_spaces(random_state: int) -> Dict[str, Tuple]:
    """Return estimators and their hyperparameter search spaces."""

    return {
        "log_reg": (
            LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state),
            {
                "C": np.logspace(-3, 2, 20),
                "penalty": ["l1", "l2"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(n_jobs=-1, random_state=random_state),
            {
                "n_estimators": [200, 400, 600, 800],
                "max_depth": [None, 8, 12, 16, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
        ),
        "grad_boost": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.8, 1.0],
            },
        ),
    }


def _create_search(
    estimator, param_grid: Dict[str, list], *, random_state: int, n_iter: int = 15
) -> RandomizedSearchCV:
    """Wrap an estimator with preprocessing and hyperparameter search."""

    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])

    param_distribution = {f"model__{key}": value for key, value in param_grid.items()}
    total_combinations = math.prod(len(values) for values in param_grid.values())
    iterations = min(n_iter, total_combinations)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distribution,
        n_iter=iterations,
        scoring="roc_auc",
        n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        verbose=1,
        random_state=random_state,
    )
    return search


def _evaluate(best_estimator, X_test, y_test) -> Dict:
    """Compute evaluation metrics."""

    y_pred = best_estimator.predict(X_test)
    y_proba = best_estimator.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def train_and_select_best(random_state: int = 42) -> Dict:
    """Train all models, evaluate on the hold-out test set, and persist the best."""

    raw = load_dataset()
    engineered = engineer_features(raw)
    dataset = train_test_split_features(engineered)

    best_model_name = None
    best_score = -np.inf
    best_estimator = None
    best_metrics: Dict = {}

    for name, (estimator, search_space) in _model_spaces(random_state).items():
        print(f"Training {name} ...")
        search = _create_search(estimator, search_space, random_state=random_state)
        search.fit(dataset.X_train, dataset.y_train)
        metrics = _evaluate(search.best_estimator_, dataset.X_test, dataset.y_test)
        print(f"{name} ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model_name = name
            best_estimator = search.best_estimator_
            best_metrics = metrics

    assert best_estimator is not None, "No model was trained successfully."

    joblib.dump(best_estimator, SAVED_MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps({"model_name": best_model_name, **best_metrics}, indent=2)
    )

    print(f"Best model: {best_model_name} -> saved to {SAVED_MODEL_PATH}")
    return {"model_name": best_model_name, **best_metrics}


if __name__ == "__main__":
    train_and_select_best()
