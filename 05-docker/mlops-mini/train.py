"""Minimal ML training script that logs to MLflow and saves a sklearn model.
Runs on the diabetes regression toy dataset for quick iterations.
"""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "diabetes-regression")
RANDOM_STATE = 42


def load_data():
    """Load the diabetes dataset as a pandas DataFrame/Series."""
    dataset = load_diabetes(as_frame=True)
    return dataset.data, dataset.target


def train_and_log():
    """Train a small ElasticNet model and log everything to MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    alpha = 0.05
    l1_ratio = 0.5
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="elasticnet-diabetes"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_params({"alpha": alpha, "l1_ratio": l1_ratio})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        signature = mlflow.models.infer_signature(X_train, preds)
        # Log the sklearn model with a minimal input example for easier serving
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2),
            registered_model_name="DiabetesElasticNet",
        )

        print("Logged run to MLflow")
        print(f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")


if __name__ == "__main__":
    train_and_log()
