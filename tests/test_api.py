"""
test_api.py
===========
Tests automatisés de l'API de détection de fraude.

En entreprise, ces tests tournent automatiquement à CHAQUE push de code
(via GitHub Actions, voir .github/workflows/ci.yml) pour éviter qu'un
changement casse l'API sans que personne ne s'en rende compte.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_schema(client):
    payload = {
        "user_id": 1,
        "amount": 100.0,
        "hour": 14,
        "merchant_category": "grocery",
        "is_foreign": 0,
        "days_since_last_txn": 1.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud_flagged"], bool)
    assert body["risk_level"] in {"low", "medium", "high", "critical"}


def test_predict_rejects_invalid_amount(client):
    """Le montant doit être strictement positif (gt=0 dans le schéma Pydantic)."""
    payload = {
        "user_id": 1,
        "amount": -50.0,
        "hour": 14,
        "merchant_category": "grocery",
        "is_foreign": 0,
        "days_since_last_txn": 1.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # erreur de validation Pydantic


def test_predict_rejects_invalid_hour(client):
    payload = {
        "user_id": 1,
        "amount": 100.0,
        "hour": 25,  # invalide : doit être entre 0 et 23
        "merchant_category": "grocery",
        "is_foreign": 0,
        "days_since_last_txn": 1.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_unknown_user_id_does_not_crash(client):
    """Un utilisateur jamais vu à l'entraînement doit quand même recevoir
    une prédiction (via les valeurs par défaut), pas une erreur serveur."""
    payload = {
        "user_id": 999999,
        "amount": 100.0,
        "hour": 14,
        "merchant_category": "grocery",
        "is_foreign": 0,
        "days_since_last_txn": 1.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
