# --- Étape 1 : image de base légère avec Python ---
FROM python:3.12-slim

# Empêche Python de créer des fichiers .pyc et force l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# On copie d'abord SEULEMENT requirements.txt pour profiter du cache Docker :
# si le code change mais pas les dépendances, Docker ne réinstalle pas tout.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# On copie ensuite le reste du code et le modèle déjà entraîné
COPY src/ ./src/
COPY models/ ./models/

WORKDIR /app/src

EXPOSE 8000

# healthcheck : Docker/Kubernetes peuvent vérifier automatiquement que l'API répond
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
