# Mini workflow MLOps local (MLflow + FastAPI + Docker Compose)

Objectif : montrer un pipeline local simple : tracking MLflow, entraînement scikit-learn, API FastAPI qui charge le dernier modèle MLflow. Tout tourne avec Docker Compose, pas de cloud.

## Lancer la stack

```bash
cd 05-docker/mlops-mini
# Build images et lancer les services (MLflow, trainer, API)
docker compose up --build
```

- MLflow UI : http://localhost:5000
- API FastAPI : http://localhost:8000/docs

## Flux pédagogique

1. `mlflow` démarre un serveur local (backend SQLite + artefacts sur volume partagé `mlruns`).
2. `trainer` lance `train.py`, entraîne un ElasticNet sur le dataset diabète sklearn, log les métriques et le modèle dans MLflow.
3. `api` démarre FastAPI, récupère le run avec le meilleur `rmse` et expose `/predict`.

## Tester l'API

Après que le training a terminé (quelques secondes) :

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235, -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613]
    ]
  }'
```

## Structure des services

- `docker-compose.yml` : orchestre 3 services et 2 volumes (`mlruns`, `mlflow_db`).
- `Dockerfile.mlflow` : image légère avec `mlflow` pour le tracking UI.
- `Dockerfile.train` + `train.py` : script simple scikit-learn qui log modèle et métriques dans MLflow.
- `Dockerfile.api` + `app.py` : FastAPI charge le meilleur run (RMSE minimal) depuis MLflow et sert `/predict`.

## Astuces / Nettoyage

- Arrêt : `docker compose down`
- Suppression volumes (réinitialise runs et artefacts) : `docker compose down -v`
