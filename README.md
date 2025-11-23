# Fraud Detection ML Project

This directory contains a complete mini-project that turns the provided fraud dataset into an end-to-end machine learning solution. The workflow covers the following checkpoints from the assignment:

1. **Problem framing** – `problem_description.py` explains the business need and why ML is a good fit.
2. **Data preparation + EDA** – `eda.py` loads `transactions.csv`, engineers time/geography features, and stores statistics + plots inside `artifacts/eda/`.
3. **Model training & tuning** – `train_models.py` evaluates Logistic Regression, Random Forest, and Gradient Boosting with randomized hyper-parameter search, then persists the champion model to `artifacts/models/best_model.joblib`.444. **Serving & Docker** – `service.py` exposes the trained model with FastAPI and the included `Dockerfile` lets you deploy it in a container.

Everything is orchestrated through the lightweight CLI in `main.py`.

## Project layout

```
ML Task/
├── artifacts/               # Generated plots, statistics, and model artifacts
├── config.py                # Shared paths and constants
├── data_pipeline.py         # Feature engineering and preprocessing builders
├── eda.py                   # Reproducible exploratory data analysis script
├── main.py                  # CLI entry point (describe / eda / train / export)
├── problem_description.py   # Narrative about the problem & ML impact
├── requirements.txt         # Python dependencies
├── service.py               # FastAPI inference service
├── train_models.py          # Model selection and persistence logic
├── transactions.csv         # Provided dataset
└── Dockerfile               # Container image for the FastAPI app
```

## Setup

```bash
cd "ML Task"
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the steps

```bash
# Describe the problem
python main.py describe

# Generate EDA artifacts in artifacts/eda/
python main.py eda

# Train, tune, and save the best model
python main.py train

# Export the supplied notebook to artifacts/exports/notebook_export.py
python main.py export
```

After training you will have:

- `artifacts/models/best_model.joblib` – serialized sklearn pipeline
- `artifacts/models/best_model_metrics.json` – ROC-AUC, PR-AUC, classification report, and confusion matrix for the hold-out test set

## Serving locally

```bash
uvicorn service:app --reload --port 8000
```

Send a prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

The payload must include the same fields that appear in `transactions.csv` (excluding `is_fraud`). The service handles feature engineering internally so you can pass raw transaction data.

## Docker deployment

```bash
# Build the container image (run from inside ML Task/)
docker build -t fraud-service .

# Ensure the model artifact exists, then launch the API
docker run -p 8000:8000 -v $(pwd)/artifacts/models:/app/artifacts/models fraud-service
```

Mounting `artifacts/models` into the container keeps the trained model accessible without rebuilding the image every time you retrain.
