# Fraud Detection MLOps Pipeline

Système complet de détection de fraude bancaire : entraînement d'un modèle
XGBoost, suivi d'expériences avec MLflow, API de scoring en temps réel avec
FastAPI, tests automatisés, monitoring de dérive des données, et pipeline
CI/CD avec GitHub Actions, le tout containerisé avec Docker.

---

## Structure du projet

```
fraud_mlops/
├── data/
│   ├── generate_data.py      # génère le dataset synthétique
│   └── transactions.csv      # généré
├── src/
│   ├── features.py           # feature engineering
│   ├── train.py               # entraînement + suivi MLflow
│   ├── api.py                  # API FastAPI de scoring
│   └── monitor.py            # détection de dérive des données
├── models/                    # modèle entraîné + métadonnées (généré)
├── tests/
│   └── test_api.py            # tests automatisés (pytest)
├── .github/workflows/ci.yml   # pipeline CI/CD
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation 

### 1. Prérequis
- Python 3.10+ installé (vérifie avec `python3 --version`)
- Git installé
- Docker Desktop installé

### 2. Cloner/télécharger le projet et se placer dedans
```bash
cd fraud_mlops
```

### 3. Créer un environnement virtuel (isole les dépendances de ce projet)
```bash
python3 -m venv venv

# Activer l'environnement virtuel
# Sur Mac/Linux :
source venv/bin/activate
# Sur Windows :
venv\Scripts\activate
```
Tu verras `(venv)` apparaître au début de ta ligne de commande, pour savoir que c'est activé.

### 4. Installer toutes les dépendances
```bash
pip install -r requirements.txt
```

---

## Utilisation pas à pas

### Étape A — Générer les données
```bash
python data/generate_data.py
```
Ça crée `data/transactions.csv` avec 150 000 transactions simulées et
affiche le taux de fraude. Regarde le fichier CSV généré pour comprendre
la structure des données.

### Étape B — Entraîner le modèle
```bash
cd src
python train.py
```
Ça va :
1. Charger les données et faire le feature engineering
2. Entraîner un modèle XGBoost
3. Afficher les métriques (precision, recall, F1, ROC-AUC)
4. Sauvegarder le modèle dans `models/model.pkl`
5. Enregistrer l'expérience dans MLflow

### Étape C — Visualiser les expériences avec MLflow
```bash
# toujours depuis le dossier src/
mlflow ui
```
Ouvre ton navigateur sur **http://localhost:5000** — tu verras toutes tes
expériences, avec les métriques comparées visuellement. C'est l'outil que
les data scientists utilisent en entreprise pour comparer différentes
versions de modèles avant de choisir laquelle déployer.

### Étape D — Lancer l'API
```bash
# toujours depuis src/
uvicorn api:app --reload --port 8000
```
Ouvre **http://localhost:8000/docs** — tu as une interface interactive
(Swagger) pour tester l'API directement dans le navigateur, sans écrire de
code. Clique sur `/predict`, "Try it out", modifie les valeurs, "Execute".

Ou teste depuis un terminal :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 42,
    "amount": 2500.75,
    "hour": 3,
    "merchant_category": "jewelry",
    "is_foreign": 1,
    "days_since_last_txn": 15.2
  }'
```

### Étape E — Lancer les tests automatisés
```bash
# depuis la racine du projet
pytest tests/ -v
```
Tu dois voir 5 tests passer. Ces tests vérifient que l'API répond
correctement, rejette les données invalides, et ne plante pas sur des cas
limites (utilisateur inconnu, etc...).

### Étape F — Vérifier la détection de dérive
```bash
cd src
python monitor.py
```
Ça montre une démonstration : comparaison de données similaires (pas de
dérive) vs données volontairement "décalées" (dérive détectée avec le
score PSI).

### Étape G — Containeriser avec Docker
```bash
# depuis la racine du projet (là où est le Dockerfile)
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```
L'API tourne maintenant dans un container isolé, exactement comme elle
tournerait en production.

```

---


