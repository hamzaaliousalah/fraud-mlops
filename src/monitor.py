"""
monitor.py
==========
Détecte la "dérive des données" (data drift) : quand les nouvelles
transactions qui arrivent en production commencent à avoir des
caractéristiques statistiques différentes de celles utilisées à
l'entraînement.

Pourquoi c'est important : un modèle entraîné sur les habitudes de 2025
devient progressivement moins fiable si les comportements changent (nouvelle
méthode de fraude, inflation qui change les montants moyens, etc.). En
entreprise, on ne réentraîne pas "au hasard" -> on surveille la dérive et on
réentraîne quand elle dépasse un seuil.

On utilise ici le Population Stability Index (PSI), une métrique standard
de l'industrie bancaire/assurance pour quantifier la dérive d'une variable :
  PSI < 0.1  -> pas de dérive significative
  0.1 - 0.25 -> dérive modérée, à surveiller
  > 0.25     -> dérive importante, réentraînement recommandé
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"


def population_stability_index(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calcule le PSI entre une distribution de référence (baseline, à
    l'entraînement) et une distribution actuelle (nouvelles données)."""
    breakpoints = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
    current_counts, _ = np.histogram(current, bins=breakpoints)

    baseline_pct = np.clip(baseline_counts / len(baseline), 1e-6, None)
    current_pct = np.clip(current_counts / len(current), 1e-6, None)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def check_drift(new_data: pd.DataFrame, reference_data: pd.DataFrame,
                 columns: list[str] | None = None) -> dict:
    """Compare de nouvelles données à des données de référence colonne par
    colonne et retourne un rapport de dérive."""
    if columns is None:
        columns = [c for c in ["amount", "hour", "days_since_last_txn"] if c in new_data.columns]

    report = {}
    for col in columns:
        psi = population_stability_index(
            reference_data[col].dropna().values,
            new_data[col].dropna().values,
        )
        if psi < 0.1:
            status = "stable"
        elif psi < 0.25:
            status = "moderate_drift"
        else:
            status = "significant_drift"

        report[col] = {"psi": round(psi, 4), "status": status}

    return report


if __name__ == "__main__":
    # Démo : on compare les données d'entraînement à elles-mêmes (PSI ~0,
    # attendu) puis à une version "shiftée" artificiellement pour montrer
    # que la détection fonctionne.
    df = pd.read_csv(ROOT / "data" / "transactions.csv")
    reference = df.sample(frac=0.5, random_state=1)
    similar_new_data = df.drop(reference.index).sample(frac=1.0, random_state=2)

    print("=== Cas 1 : nouvelles données similaires (pas de dérive attendue) ===")
    print(json.dumps(check_drift(similar_new_data, reference), indent=2))

    # On simule une dérive réelle : les montants doublent (ex: inflation,
    # ou nouveau segment de clients à gros montants)
    drifted_data = similar_new_data.copy()
    drifted_data["amount"] = drifted_data["amount"] * 2.5

    print("\n=== Cas 2 : montants qui ont dérivé (dérive attendue) ===")
    print(json.dumps(check_drift(drifted_data, reference), indent=2))
