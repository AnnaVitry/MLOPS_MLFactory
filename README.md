# 🌸 ML Factory : Infrastructure Iris MLOps

Ce projet implémente une usine MLOps complète pour le classement du dataset Iris. Il intègre le tracking d'expériences, un registre de modèles, un stockage d'artefacts S3 et une API capable de recharger les modèles "à chaud" sans interruption de service.

## Arborescence du Projet

```bash
MLOPS_MLFACTORY/
├── data/                   # Données générées pour les tests (partagé via volume Docker)
│   └── iris_test.csv       # Fichier CSV utilisé par le Front-end
├── src/
│   ├── api/                # Backend FastAPI
│   │   ├── main.py         # Logique de rechargement à chaud et inférence
│   │   └── Dockerfile      # Image pour l'API
│   ├── front/              # Interface Streamlit
│   │   ├── app.py          # Visualisation et interaction utilisateur
│   │   └── Dockerfile      # Image pour le Front-end
│   └── train/              # Pipeline d'entraînement
│       └── train.py        # Entraînement, log MLflow et gestion des alias
├── .env                    # Variables d'environnement (Secrets et URLs)
├── docker-compose.yml      # Orchestration des services (MinIO, MLflow, API, Front)
├── generate_data.py        # Script de création du dataset de test
└── pyproject.toml          # Gestion des dépendances avec uv
```

---

## Commandes de Pilotage

### 1. Démarrage de l'infrastructure
Pour lancer tous les services (MinIO, MLflow, API, Front) :
```bash
docker compose up -d --build
```

### 2. Gestion des données
Pour générer un fichier `iris_test.csv` avec un échantillonnage aléatoire :
```bash
uv run generate_data.py
```

### 3. Cycle d'entraînement (CI/CD ML)
Chaque exécution crée une nouvelle version du modèle dans MLflow :
```bash
uv run src/train/train.py
```

### 4. Maintenance et Nettoyage Radical
En cas de problème de daemon ou pour tout remettre à zéro :
```bash
docker compose down -v  # Supprime tout, y compris les volumes (données MinIO)
docker ps               # Vérifie que les 4 conteneurs sont "Up"
```
 
Si ton environnement Docker devient instable ou si tu manques d'espace disque, utilise ces commandes. 

> [!WARNING]
> **Attention :** Le nettoyage des volumes supprimera toutes les données persistantes (historique MLflow et fichiers dans MinIO) qui ne sont pas activement liées à un conteneur en cours d'exécution.  
```bash
systemctl --user restart docker-desktop # Relance le moteur Docker Desktop (Utile en cas d'erreur "docker.sock")
docker builder prune -a --force # Supprime l'intégralité du cache de build (Force une reconstruction "neuve" des images)
docker image prune -a -f # Supprime toutes les images non utilisées (Nettoyage massif du stockage)
docker volume prune -f # Supprime les volumes orphelins (Efface les données MinIO/MLflow non actives) 
docker container prune -f # Supprime tous les conteneurs arrêtés
```
---

## Procédure de Test des Modèles (A/B Testing)

Le script `src/train/train.py` est conçu pour basculer facilement entre deux algorithmes.

### Comment changer de modèle ?
Ouvre `src/train/train.py` et localise le bloc de sélection du modèle :

#### **Option A : Régression Logistique (Modèle Linéaire)**
```python
# Décommentez ces lignes :
model = LogisticRegression(max_iter=200)
model_type = "LogisticRegression"

# Commentez ces lignes :
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model_type = "RandomForest"
```

#### **Option B : Random Forest (Modèle d'Ensemble)**
```python
# Commentez ces lignes :
# model = LogisticRegression(max_iter=200)
# model_type = "LogisticRegression"

# Décommentez ces lignes :
model = RandomForestClassifier(n_estimators=100, random_state=42)
model_type = "RandomForest"
```
et décommenter toutes les lignes qui ont le commentaire "# Décommenter pour RandomTree". Obviously \*deadass, with a Severus Snape voice\*)

**Note sur la traçabilité :** Pense à modifier `n_estimators` pour créer des versions (V1, V2, V3) et observer le changement dynamique sur l'interface.

---

## Fonctionnement du Rechargement à Chaud (Hot Reload)

L'API (`src/api/main.py`) utilise un mécanisme de cache intelligent pour éviter les redémarrages manuels :

1. **Vérification de l'Alias** : À chaque requête de prédiction, l'API demande à MLflow : *"Quelle version porte l'alias 'production' ?"*.
2. **Comparaison de Version** : 
   - Si `prod_version == state["version"]`, l'API utilise le modèle déjà en mémoire (ultra-rapide).
   - Si `prod_version != state["version"]`, l'API télécharge automatiquement le nouveau fichier `.pkl` depuis MinIO.
3. **Mise à jour Front** : L'interface Streamlit récupère la clé `model_version` renvoyée par l'API et met à jour l'affichage instantanément.

---

## Accès aux Services

| Service | URL | Usage |
| :--- | :--- | :--- |
| **Streamlit UI** | `http://localhost:8501` | Interface utilisateur finale |
| **MLflow UI** | `http://localhost:5000` | Suivi des runs et registre de modèles |
| **MinIO Console** | `http://localhost:9001` | Exploration des fichiers (Artefacts) |
| **FastAPI Docs** | `http://localhost:8000/docs` | Documentation interactive de l'API |

---

## Troubleshooting (Dépannage)

* **Erreur 404 sur /predict** : 
  - Vérifie dans MLflow si l'alias `production` est bien assigné à une version.
  - Vérifie la casse dans ton `.env` : `MODEL_ALIAS=production` (en minuscules).
* **Erreur Connection Refused (9000)** :
  - Le conteneur MinIO n'est pas prêt. Attends 10 secondes ou vérifie `docker ps`.
* **Les changements de code ne s'affichent pas** :
  - Si tu n'utilises pas les volumes dans Docker Compose, tu dois reconstruire l'image : `docker compose up -d --build front`.

---
*Ce document doit être mis à jour à chaque ajout de nouvelle fonctionnalité (ex: monitoring, tests unitaires).*