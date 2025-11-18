"""Exploratory data analysis helpers."""

from __future__ import annotations

import json
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import EDA_DIR
from data_pipeline import engineer_features, load_dataset


def _save_plot(fig: plt.Figure, name: str) -> None:
    """Persist a matplotlib figure inside the artifacts folder."""

    path = EDA_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _target_distribution(df: pd.DataFrame) -> Dict[str, float]:
    counts = df["is_fraud"].value_counts(normalize=True).sort_index()
    return {str(idx): round(float(value), 6) for idx, value in counts.items()}


def run_eda() -> None:
    """Run a lightweight but reproducible set of analyses."""

    raw = load_dataset()
    engineered = engineer_features(raw)

    summary = engineered.describe(include="all")
    summary.to_csv(EDA_DIR / "summary_statistics.csv")

    target_stats = _target_distribution(engineered)
    (EDA_DIR / "target_distribution.json").write_text(json.dumps(target_stats, indent=2))

    categorical_breakdown = (
        engineered.groupby("channel")["is_fraud"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    categorical_breakdown.to_csv(EDA_DIR / "channel_fraud_rate.csv")

    country_risk = (
        engineered.groupby("country")["is_fraud"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    country_risk.to_csv(EDA_DIR / "top_country_risk.csv")

    corr = engineered[
        [
            "amount",
            "avg_amount_user",
            "shipping_distance_km",
            "transaction_hour",
            "transaction_day",
            "amount_vs_avg",
            "total_transactions_user",
            "is_fraud",
        ]
    ].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    _save_plot(fig, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(engineered, x="amount", hue="is_fraud", bins=40, ax=ax, stat="density", kde=True)
    ax.set_title("Transaction Amount Distribution by Class")
    _save_plot(fig, "amount_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=engineered, x="channel", y="amount_vs_avg", hue="is_fraud", ax=ax)
    ax.set_title("Amount vs. User Average by Channel")
    _save_plot(fig, "amount_vs_avg_by_channel.png")

    print("EDA artifacts written to", EDA_DIR)


if __name__ == "__main__":
    run_eda()
