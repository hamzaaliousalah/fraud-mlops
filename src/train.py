"""
train.py
========
Entraîne un modèle XGBoost de détection de fraude, évalue sa performance
avec les BONNES métriques (pas juste l'accuracy, qui est trompeuse sur des
données déséquilibrées), et enregistre tout dans MLflow.

Pourquoi pas juste "accuracy" ?
Si 1.5% des transactions sont frauduleuses, un modèle qui dit TOUJOURS
"pas de fraude" a 98.5% d'accuracy... et 0% d'utilité. On utilise donc :
- Precision : parmi les transactions signalées comme fraude, combien le sont vraiment ?
- Recall : parmi les vraies fraudes, combien le modèle en détecte ?
- ROC-AUC et PR-AUC : mesure globale de la capacité à séparer les classes
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from xgboost import XGBClassifier

from features import engineer_features, get_feature_columns

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "transactions.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_and_split():
    df = pd.read_csv(DATA_PATH)

    # Split temporel-like : ici aléatoire car données synthétiques,
    # mais en prod avec de vraies données, on splitterait PAR DATE
    # (entraîner sur le passé, tester sur le futur) pour éviter la fuite.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_fraud"]
    )
    return train_df, test_df


def train_model():
    train_df, test_df = load_and_split()

    # Feature engineering : on "fit" sur le train, on applique sur le test
    train_feat, user_stats = engineer_features(train_df, fit=True)
    test_feat, _ = engineer_features(test_df, fit=False, user_stats=user_stats)

    feature_cols = get_feature_columns(train_feat)
    # S'assurer que test a exactement les mêmes colonnes one-hot que train
    for col in feature_cols:
        if col not in test_feat.columns:
            test_feat[col] = 0
    test_feat = test_feat[feature_cols + ["is_fraud"]]

    X_train, y_train = train_feat[feature_cols], train_feat["is_fraud"]
    X_test, y_test = test_feat[feature_cols], test_feat["is_fraud"]

    # scale_pos_weight compense le déséquilibre de classes : on dit au modèle
    # "une fraude manquée coûte beaucoup plus cher qu'une fausse alerte"
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        mlflow.log_metrics({
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        })

        mlflow.xgboost.log_model(model, artifact_path="model")

        # On sauvegarde aussi une copie locale simple pour l'API
        # (plus simple à charger que d'aller chercher dans MLflow en prod légère)
        joblib.dump(model, MODEL_DIR / "model.pkl")
        joblib.dump(user_stats, MODEL_DIR / "user_stats.pkl")
        joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")

        # Statistiques de référence pour la détection de dérive (étape monitoring)
        baseline_stats = {
            col: {"mean": float(X_train[col].mean()), "std": float(X_train[col].std())}
            for col in feature_cols
        }
        with open(MODEL_DIR / "baseline_stats.json", "w") as f:
            json.dump(baseline_stats, f, indent=2)

        print(f"Run MLflow ID : {run.info.run_id}")
        print("Métriques :")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"Matrice de confusion -> TP={tp} FP={fp} TN={tn} FN={fn}")

    return model, metrics


if __name__ == "__main__":
    train_model()
