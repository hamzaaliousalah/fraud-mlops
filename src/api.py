"""
api.py
======
Expose le modèle entraîné via une API REST avec FastAPI.

Deux endpoints :
- GET  /health   -> vérifie que l'API et le modèle sont bien chargés
- POST /predict  -> reçoit une transaction, retourne une probabilité de fraude

Lance avec : uvicorn api:app --reload --port 8000
Documentation interactive auto-générée sur : http://localhost:8000/docs
"""

from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import engineer_features, get_feature_columns

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"

# Dictionnaire global qui contiendra le modèle et ses dépendances une fois chargés
ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ce code s'exécute UNE SEULE FOIS au démarrage de l'API (pas à chaque requête)
    # -> évite de recharger le modèle depuis le disque à chaque appel, ce qui serait
    # beaucoup trop lent en production.
    ml_artifacts["model"] = joblib.load(MODEL_DIR / "model.pkl")
    ml_artifacts["user_stats"] = joblib.load(MODEL_DIR / "user_stats.pkl")
    ml_artifacts["feature_cols"] = joblib.load(MODEL_DIR / "feature_cols.pkl")
    print("Modèle chargé avec succès.")
    yield
    ml_artifacts.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="API de scoring de fraude en temps réel pour transactions financières",
    version="1.0.0",
    lifespan=lifespan,
)


class Transaction(BaseModel):
    """Schéma d'une transaction entrante. Pydantic valide automatiquement
    les types et génère la documentation Swagger."""
    user_id: int = Field(..., json_schema_extra={"example": 42})
    amount: float = Field(..., gt=0, json_schema_extra={"example": 250.75})
    hour: int = Field(..., ge=0, le=23, json_schema_extra={"example": 3})
    merchant_category: str = Field(..., json_schema_extra={"example": "jewelry"})
    is_foreign: int = Field(..., ge=0, le=1, json_schema_extra={"example": 1})
    days_since_last_txn: float = Field(..., ge=0, json_schema_extra={"example": 15.2})


class PredictionResponse(BaseModel):
    transaction_id_echo: int | None = None
    fraud_probability: float
    is_fraud_flagged: bool
    risk_level: str
    decision_threshold: float


DECISION_THRESHOLD = 0.5  # ajustable selon le compromis precision/recall voulu


def _risk_level(proba: float) -> str:
    if proba < 0.2:
        return "low"
    elif proba < 0.5:
        return "medium"
    elif proba < 0.8:
        return "high"
    return "critical"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in ml_artifacts,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if "model" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # On reconstruit un DataFrame à une ligne avec la même structure
    # que les données d'entraînement, pour appliquer EXACTEMENT le même
    # pipeline de feature engineering.
    raw = pd.DataFrame([transaction.model_dump()])

    feat, _ = engineer_features(
        raw, fit=False, user_stats=ml_artifacts["user_stats"]
    )

    feature_cols = ml_artifacts["feature_cols"]
    for col in feature_cols:
        if col not in feat.columns:
            feat[col] = 0
    feat = feat[feature_cols]

    model = ml_artifacts["model"]
    proba = float(model.predict_proba(feat)[:, 1][0])

    return PredictionResponse(
        fraud_probability=round(proba, 4),
        is_fraud_flagged=proba >= DECISION_THRESHOLD,
        risk_level=_risk_level(proba),
        decision_threshold=DECISION_THRESHOLD,
    )
