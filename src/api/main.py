import os

import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

# Dans src/api/main.py


load_dotenv()  # Charge les variables du .env

app = FastAPI(title="ML Factory API")

# Configuration via variables d'environnement
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "iris_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# Initialisation du client
client = MlflowClient(tracking_uri=MLFLOW_URI)

# Cache en mémoire pour éviter de recharger si la version n'a pas changé [cite: 101]
state = {"model": None, "version": None}


class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def load_production_model():
    """Vérifie l'alias 'Production' et recharge si nécessaire [cite: 105, 106]"""
    try:
        # 1. On demande quelle est la version actuelle de l'alias 'Production' [cite: 108]
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        prod_version = alias_info.version

        # 2. Si le modèle n'est pas en cache ou si la version a changé [cite: 109]
        if state["model"] is None or prod_version != state["version"]:
            print(f"🔄 Rechargement à chaud : Passage à la version {prod_version}")
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            state["model"] = mlflow.pyfunc.load_model(model_uri)
            state["version"] = prod_version

        return state["model"], state["version"]
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur MLflow: {str(e)}")


@app.post("/predict")
def predict(data: IrisData):
    # Récupération dynamique du modèle
    model, version = load_production_model()

    # Préparation des données pour scikit-learn
    features = [
        [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    ]

    # Inférence
    prediction = model.predict(features)

    # On renvoie la prédiction ET la version pour la traçabilité
    return {
        "prediction": int(prediction[0]),
        "model_version": version,
        "status": "success",
    }


@app.get("/health")
def health():
    return {"status": "up"}
