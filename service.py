"""FastAPI service that hosts the trained fraud detection model."""

from __future__ import annotations

from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import SAVED_MODEL_PATH
from data_pipeline import engineer_features


class TransactionPayload(BaseModel):
    user_id: int
    account_age_days: float
    total_transactions_user: int
    avg_amount_user: float
    amount: float
    country: str
    bin_country: str
    channel: str
    merchant_category: str
    promo_used: int = Field(..., ge=0, le=1)
    avs_match: int = Field(..., ge=0, le=1)
    cvv_result: int = Field(..., ge=0, le=1)
    three_ds_flag: int = Field(..., ge=0, le=1)
    transaction_time: str = Field(..., description="ISO 8601 timestamp")
    shipping_distance_km: float
    transaction_id: Optional[int] = None


app = FastAPI(title="Fraud Detection Service", version="1.0.0")


def _load_model():
    if not SAVED_MODEL_PATH.exists():
        raise RuntimeError(
            f"Missing model artifact at {SAVED_MODEL_PATH}. Run train_models.py first."
        )
    return joblib.load(SAVED_MODEL_PATH)


try:
    MODEL = _load_model()
    MODEL_ERROR = None
except RuntimeError as exc:
    MODEL = None
    MODEL_ERROR = str(exc)


@app.get("/health")
def healthcheck():
    if MODEL_ERROR:
        return {"status": "degraded", "detail": MODEL_ERROR}
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TransactionPayload):
    if MODEL is None:
        raise HTTPException(status_code=503, detail=MODEL_ERROR or "Model unavailable.")

    frame = pd.DataFrame([payload.dict()])
    features = engineer_features(frame)
    features = features.drop(columns=["is_fraud"], errors="ignore")

    fraud_probability = float(MODEL.predict_proba(features)[0][1])
    prediction = int(fraud_probability >= 0.5)

    return {
        "fraud_probability": fraud_probability,
        "prediction": prediction,
        "threshold": 0.5,
    }
