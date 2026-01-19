"""FastAPI inference service that fetches the best MLflow run and serves /predict."""

import os
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, conlist

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "diabetes-regression")
FEATURE_NAMES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

app = FastAPI(title="MLflow + FastAPI demo", version="0.1.0")
client = MlflowClient(tracking_uri=TRACKING_URI)
model = None
model_metrics: Optional[dict] = None


class PredictRequest(BaseModel):
    # Enforce 10 float features per sample (diabetes dataset)
    features: list[conlist(float, min_length=10, max_length=10)]


@app.on_event("startup")
def load_model_on_startup():
    """Load the best (lowest RMSE) run from MLflow once at startup."""
    global model, model_metrics
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run training first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found in MLflow. Run training first.")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    model_metrics = best_run.data.metrics


@app.get("/health")
def healthcheck():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    inputs = np.array(payload.features)
    if inputs.shape[1] != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail="Each sample must have 10 features")

    input_df = pd.DataFrame(inputs, columns=FEATURE_NAMES)
    # Use DataFrame to satisfy MLflow model signature column names
    preds = model.predict(input_df)
    return {"predictions": preds.tolist(), "metrics_used_for_selection": model_metrics}
