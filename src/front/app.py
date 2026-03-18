import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="ML Factory", layout="wide")

# Paramètres
API_URL = os.getenv("API_URL", "http://api:8000/predict")
DATA_PATH = "data/iris_test.csv"

st.title("🌸 ML Factory : Inférence Zero-Downtime")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    sample_idx = st.sidebar.number_input("Choisir une ligne", 0, len(df) - 1, 0)
    row = df.iloc[sample_idx]

    # Utilisation des nouveaux noms sans (cm)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sepal Length cm", row["sepal_length"])
    col2.metric("Sepal Width cm", row["sepal_width"])
    col3.metric("Petal Length cm", row["petal_length"])
    col4.metric("Petal Width cm", row["petal_width"])

    if st.button("🚀 Lancer la prédiction"):
        payload = {
            "sepal_length": float(row["sepal_length"]),
            "sepal_width": float(row["sepal_width"]),
            "petal_length": float(row["petal_length"]),
            "petal_width": float(row["petal_width"]),
        }
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()

            prediction_idx = result.get("prediction")
            target_names = ["Setosa", "Versicolor", "Virginica"]
            flower_name = target_names[prediction_idx]

            version = result.get("model_version", "Inconnue")
            st.success(f"### Résultat : {result.get('prediction')} = {flower_name}")
            st.info(f"Modèle utilisé : Version {version}")
        except Exception as e:
            st.error(f"Erreur API : {e}")
else:
    st.error("Générez d'abord le CSV avec underscores.")
