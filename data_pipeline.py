"""Reusable data loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_PATH

CATEGORICAL_FEATURES = ["country", "bin_country", "channel", "merchant_category"]
NUMERIC_FEATURES = [
    "account_age_days",
    "total_transactions_user",
    "avg_amount_user",
    "amount",
    "shipping_distance_km",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "amount_vs_avg",
    "days_since_first",
]
BINARY_FLAG_FEATURES = [
    "promo_used",
    "avs_match",
    "cvv_result",
    "three_ds_flag",
    "is_weekend",
    "country_mismatch",
]


@dataclass
class PreparedDataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load the raw csv file."""

    df = pd.read_csv(path)
    df = df.sort_values("transaction_time")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based, ratio, and geographic features used downstream."""

    engineered = df.copy()
    engineered["transaction_time"] = pd.to_datetime(engineered["transaction_time"])
    engineered["transaction_hour"] = engineered["transaction_time"].dt.hour
    engineered["transaction_day"] = engineered["transaction_time"].dt.dayofweek
    engineered["transaction_month"] = engineered["transaction_time"].dt.month
    engineered["is_weekend"] = (engineered["transaction_day"] >= 5).astype(int)

    engineered["country_mismatch"] = (
        engineered["country"] != engineered["bin_country"]
    ).astype(int)
    engineered["amount_vs_avg"] = engineered["amount"] / (
        engineered["avg_amount_user"] + 1e-6
    )
    engineered["days_since_first"] = (
        engineered.groupby("user_id")["transaction_time"]
        .transform(lambda s: (s - s.min()).dt.total_seconds() / (24 * 3600))
        .fillna(0.0)
    )

    columns_to_drop = [
        "transaction_id",
        "transaction_time",
    ]
    engineered = engineered.drop(columns=[c for c in columns_to_drop if c in engineered.columns])
    return engineered


def train_test_split_features(
    df: pd.DataFrame, target: str = "is_fraud", test_size: float = 0.3, random_state: int = 42
) -> PreparedDataset:
    """Split the dataset into train/test components."""

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return PreparedDataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer that handles numerical scaling and categorical encoding."""

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    binary_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("binary", binary_transformer, BINARY_FLAG_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def _make_one_hot_encoder() -> OneHotEncoder:
    """Return a dense-output OneHotEncoder compatible with multiple sklearn versions."""
    kwargs = {"handle_unknown": "ignore"}
    parameters = inspect.signature(OneHotEncoder.__init__).parameters
    if "sparse_output" in parameters:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)
