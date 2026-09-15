"""
features.py
============
Transforme les données brutes de transactions en "features" (variables)
que le modèle peut exploiter.

Règle d'or en détection de fraude : le montant BRUT d'une transaction dit peu
de choses. Ce qui compte, c'est l'ÉCART par rapport au comportement HABITUEL
de l'utilisateur. C'est pour ça qu'on construit des features "relatives"
(ex: amount_vs_user_avg) plutôt que d'utiliser juste "amount".
"""

import pandas as pd
import numpy as np


CATEGORICAL_COLS = ["merchant_category"]
NUMERIC_FEATURE_COLS = [
    "amount",
    "hour",
    "is_foreign",
    "days_since_last_txn",
    "amount_vs_user_avg",
    "is_night",
    "user_txn_count",
]


def engineer_features(df: pd.DataFrame, fit: bool = True, user_stats: dict | None = None):
    """
    Ajoute des colonnes dérivées au DataFrame.

    Paramètres
    ----------
    df : DataFrame brut (sortie de generate_data.py ou nouvelles transactions)
    fit : si True, on calcule les statistiques utilisateur à partir de ce df
          (utilisé à l'entraînement). Si False, on utilise des statistiques
          déjà calculées (`user_stats`) — c'est ce qu'on fait en production,
          car on ne doit JAMAIS recalculer les stats sur les données de test
          (ça créerait une fuite de données / "data leakage").
    user_stats : dict {user_id: montant_moyen_historique} pré-calculé.

    Retourne
    --------
    df enrichi, et le dictionnaire user_stats (à sauvegarder pour la prod).
    """
    df = df.copy()

    if fit:
        user_means = df.groupby("user_id")["amount"].mean().to_dict()
        user_txn_counts = df.groupby("user_id")["amount"].count().to_dict()
    else:
        if user_stats is None:
            raise ValueError("user_stats est requis quand fit=False")
        user_means = user_stats["_means"]
        user_txn_counts = user_stats["_txn_counts"]

    global_avg_amount = df["amount"].mean()

    df["user_avg_amount"] = df["user_id"].map(user_means).fillna(global_avg_amount).astype(float)
    df["amount_vs_user_avg"] = (df["amount"] / df["user_avg_amount"].clip(lower=1.0)).astype(float)

    df["is_night"] = df["hour"].between(0, 5).astype(int)

    df["user_txn_count"] = df["user_id"].map(user_txn_counts).fillna(1).astype(int)

    if fit:
        user_stats_out = {"_means": user_means, "_txn_counts": user_txn_counts}
    else:
        user_stats_out = user_stats

    # One-hot encoding de la catégorie de marchand
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix="cat")

    return df, user_stats_out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des colonnes de features (numériques + one-hot)."""
    cat_cols = [c for c in df.columns if c.startswith("cat_")]
    return NUMERIC_FEATURE_COLS + cat_cols
