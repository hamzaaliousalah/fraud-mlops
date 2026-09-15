"""
generate_data.py
=================
Génère un dataset synthétique de transactions financières avec un
déséquilibre réaliste (fraude = événement rare, ~1-2% des transactions).

Pourquoi synthétique ? Les vrais datasets bancaires sont confidentiels.
En entreprise, tu recevrais un dataset réel — ici on simule un dataset
qui a les MÊMES caractéristiques statistiques qu'un vrai dataset de fraude :
- déséquilibre extrême entre classes
- montants qui suivent une distribution log-normale (beaucoup de petites
  transactions, quelques grosses)
- patterns temporels (fraude plus fréquente la nuit)
- comportement utilisateur (fraude = souvent un montant inhabituel pour cet
  utilisateur précis)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# On fixe la graine aléatoire pour que le dataset soit reproductible
# (important en ML : sans ça, chaque run donnerait un résultat différent)
RNG = np.random.default_rng(42)

N_USERS = 2000
N_TRANSACTIONS = 150_000
FRAUD_RATE = 0.015  # 1.5% de fraude, réaliste pour ce type de dataset

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "restaurant", "online_retail",
    "gas_station", "entertainment", "utilities", "jewelry", "cash_advance",
]


def generate_dataset(n_transactions: int = N_TRANSACTIONS) -> pd.DataFrame:
    user_ids = RNG.integers(1, N_USERS + 1, size=n_transactions)

    # Chaque utilisateur a un "profil de dépense moyen" -> sert plus tard
    # pour détecter les montants anormaux PAR utilisateur (feature clé
    # en détection de fraude : ce n'est pas le montant absolu qui compte,
    # c'est l'écart par rapport au comportement habituel de la personne).
    user_avg_spend = {u: RNG.lognormal(mean=3.5, sigma=0.6) for u in range(1, N_USERS + 1)}

    amounts = np.array([
        max(1.0, RNG.lognormal(mean=np.log(user_avg_spend[u]), sigma=0.8))
        for u in user_ids
    ])

    hours = RNG.integers(0, 24, size=n_transactions)
    merchant_category = RNG.choice(MERCHANT_CATEGORIES, size=n_transactions)
    is_foreign = RNG.random(n_transactions) < 0.08
    days_since_last_txn = RNG.exponential(scale=2.0, size=n_transactions)

    # --- Construction du label de fraude ---
    # On simule un "score de risque" latent influencé par plusieurs facteurs
    # réalistes, puis on transforme ce score en probabilité de fraude.
    risk_score = (
        0.02
        + 0.35 * (amounts > np.array([user_avg_spend[u] * 4 for u in user_ids]))
        + 0.15 * ((hours >= 0) & (hours <= 5))          # transactions nocturnes
        + 0.20 * is_foreign
        + 0.10 * (merchant_category == "cash_advance")
        + 0.10 * (merchant_category == "jewelry")
        + 0.08 * (days_since_last_txn > 10)               # compte "dormant" soudain actif
    )
    risk_score = np.clip(risk_score, 0, 0.95)

    is_fraud = (RNG.random(n_transactions) < risk_score).astype(int)

    # On force le taux global à ~FRAUD_RATE en sous-échantillonnant les fraudes
    # excédentaires (sinon notre simulation donnerait un taux trop élevé)
    fraud_idx = np.where(is_fraud == 1)[0]
    target_n_fraud = int(n_transactions * FRAUD_RATE)
    if len(fraud_idx) > target_n_fraud:
        keep = RNG.choice(fraud_idx, size=target_n_fraud, replace=False)
        is_fraud = np.zeros(n_transactions, dtype=int)
        is_fraud[keep] = 1

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_transactions + 1),
        "user_id": user_ids,
        "amount": np.round(amounts, 2),
        "hour": hours,
        "merchant_category": merchant_category,
        "is_foreign": is_foreign.astype(int),
        "days_since_last_txn": np.round(days_since_last_txn, 2),
        "is_fraud": is_fraud,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).parent / "transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset généré : {out_path}")
    print(f"Nombre de transactions : {len(df):,}")
    print(f"Taux de fraude : {df['is_fraud'].mean() * 100:.2f}%")
    print(df.head())
