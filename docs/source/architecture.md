# Architecture de la ML Factory

L'infrastructure de la ML Factory repose sur une architecture de microservices conteneurisée, assurant la séparation stricte entre l'entraînement des modèles et leur inférence en production.

## Composants Principaux

1. **MinIO (S3 Compatible)**
   * Rôle : Stockage des artefacts (Data Lake / Model Lake).
   * Tous les fichiers binaires `.pkl` générés par Scikit-Learn sont sauvegardés ici.

2. **MLflow Tracking & Registry**
   * Rôle : Cerveau MLOps.
   * Suivi des métriques (Accuracy, Loss).
   * Gestion du cycle de vie des modèles via le système d'**Alias** (ex: `production`).

3. **FastAPI (Inférence)**
   * Rôle : Servir le modèle.
   * Implémente une logique de **Hot-Reloading** : interroge MLflow à chaque requête pour s'assurer que le modèle chargé en RAM est bien la version taguée "production", évitant ainsi les redémarrages du serveur.

4. **Streamlit (Front-End)**
   * Rôle : Interface utilisateur.
   * Communique de manière asynchrone avec FastAPI.