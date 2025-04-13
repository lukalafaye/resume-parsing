# Airflow x MLflow

Ce projet présente un pipeline complet de deep learning qui intègre :
- Ingestion et prétraitement des données
- Entraînement et validation de modèle spacy
- Orchestration avec airflow
- Suivi des expérimentations et gestion des modèles avec MLflow (todo)
- Déploiement du modèle via une api rest (todo)
- Conteneurisation de l'environnement avec docker (todo)

## Prérequis

- python 3.10
- docker
- airflow
- mlflow
- git

## Installation et configuration

1. Cloner le dépôt et se placer dans le répertoire :
   `git clone https://github.com/lukalafaye/resume-parsing`
   `cd resume-parsing`
2. Créer et activer l'environnement virtuel :
   `python3 -m venv .venv`
   `source .venv/bin/activate`
3. Installer les dépendances :
   `pip install -r requirements.txt`

## Utilisation

### Airflow

Initialiser la base de données et le compte admin :
```
airflow db init
airflow users create \
  --username admin \
  --firstname FIRST_NAME \
  --lastname LAST_NAME \
  --role Admin \
  --email admin@example.com
```
      
Modifier les paramètres de `airflow/airflow.cfg`:
```
dags_folder = /chemin/vers/le/dossier/airflow_dags
load_examples = False
```

Lancer airflow: `airflow standalone` et accéder à http://localhost:8080 pour déclencher manuellement le dag.

### MLflow (todo)

Lancer MLflow avec `mlflow ui` et consulter http://localhost:5000 pour suivre les runs, paramètres, métriques et modèles.

### Docker (todo)

1. Construire l'image :
   `docker build -t airflow .`
2. Lancer un conteneur :
   `docker run -it --name projet_ml_container airflow`
   puis dans le conteneur, exécuter les scripts Python de `src/`

### Déploiement du modèle (todo)