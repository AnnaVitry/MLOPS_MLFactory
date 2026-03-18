"""Script d'entraînement et d'enregistrement des modèles dans le Model Registry."""

import os

import boto3
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris

# from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

load_dotenv()  # Charge le fichier .env situé à la racine

# Maintenant os.environ peut lire les valeurs
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

# Pour MLflow, on utilise set_tracking_uri
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

# Le reste de ton code...
model_alias = os.getenv("MODEL_ALIAS", "production")


def prepare_minio():
    """
    Initialise le stockage S3 (MinIO).
    Vérifie si le bucket 'mlflow' existe, sinon le crée automatiquement via Boto3.
    """
    s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if "mlflow" not in buckets:
        s3.create_bucket(Bucket="mlflow")
        print("✅ Bucket 'mlflow' créé avec succès.")


def train_and_register():
    """
    Entraîne un modèle RandomForest sur le dataset Iris et l'enregistre dans MLflow.
    Assigne automatiquement l'alias défini en variable d'environnement à la nouvelle version.
    """
    # Configuration du serveur de tracking
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Iris_Factory")

    # Préparation des données
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2
    )

    # --- CHOIX DU MODÈLE (Phase 1 vs Phase 2) ---
    # Phase 1 : LogisticRegression
    model = LogisticRegression(max_iter=200)
    model_type = "LogisticRegression"

    # Phase 2 : Décommentez ceci et commentez la Phase 1
    # n_trees = 200
    # model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    # model_type = "RandomForest"

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # Log des métriques
        mlflow.log_param("model_type", model_type)
        # mlflow.log_param("n_estimators", n_trees) # Décommenter pour RandomForest
        mlflow.log_metric("accuracy", accuracy)

        # Enregistrement dans MinIO ET dans le Model Registry
        model_name = "iris_model"
        model_info = mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", registered_model_name=model_name
        )
        print(
            # f"📦 Modèle {model_type} (run_id: {run_id}) enregistré.) avec l'accuracy: {accuracy:.2f} | Arbres: {n_trees}" # Décommenter pour Random Forest
            f"📦 Modèle {model_type} (run_id: {run_id}) enregistré.) avec l'accuracy: {accuracy:.2f}"  # Décommenter pour LinearRegression
        )

    # --- GESTION DE L'ALIAS 'PRODUCTION' ---
    client = MlflowClient()
    # latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    latest_version = model_info.registered_model_version
    # ATTENTION : Pour la Phase 2, commentez la ligne ci-dessous
    client.set_registered_model_alias(model_name, model_alias, str(latest_version))
    print(f"🚀 Version {latest_version} passée en alias '{model_alias}'.")


if __name__ == "__main__":
    prepare_minio()
    train_and_register()
