# scripts/train_model.py
import json
import random
import os
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import mlflow
import mlflow.spacy
from project_paths import DATA_DIR
from project_paths import MODELS_DIR
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Fonction d'évaluation du modèle ---
def evaluate_model(nlp, dataset):
    """
    Fonction pour évaluer le modèle spaCy en termes de précision, rappel et F1-score.
    """
    y_true = []
    y_pred = []
    
    for text, ann in dataset:
        doc = nlp(text)
        true_entities = [label for start, end, label in ann["entities"]]
        predicted_entities = [ent.label_ for ent in doc.ents]
        
        y_true.extend(true_entities)
        y_pred.extend(predicted_entities)
    
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return precision, recall, f1


# --- Fonction d'entraînement du modèle spaCy NER ---
def train_spacy_ner(dataset, n_iter=10, drop=0.3, mlflow_run=False):
    """
    Fonction pour entraîner le modèle spaCy pour l'annotation des entités nommées (NER).
    """
    # Création d'un modèle vierge spaCy
    nlp = spacy.blank("en")
    
    # Ajout de la pipeline NER au modèle
    if "ner" not in nlp.pipe_names:
        # Créer le composant NER avec le nom 'ner'
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    # Ajout des entités au pipeline
    for _, annotations in dataset:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])  # Ajouter l'étiquette de l'entité
    
    # Début de l'entraînement
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        print(f"Training iteration {itn+1} of {n_iter}")
        random.shuffle(dataset)
        losses = {}
        # Entraîner sur chaque exemple
        for text, annotations in dataset:
            doc = nlp.make_doc(text)
            example = spacy.training.Example.from_dict(doc, annotations)
            nlp.update([example], drop=drop, losses=losses)
        
        print(f"Losses at iteration {itn+1}: {losses}")
    
    return nlp


# --- Fonction de logique du training ---
def train_model_logic(**context):
    """
    Airflow PythonOperator entry point for training.
    - Lit le fichier preprocessed_data.json
    - Entraîne le modèle spaCy
    - Sauvegarde localement et logue avec MLflow
    """
    ti = context['ti']
    preprocessed_path = ti.xcom_pull(key='train_data_path', task_ids='preprocess_data')

    model_out_dir = os.path.join(MODELS_DIR, "my_spacy_model")
    os.makedirs(model_out_dir, exist_ok=True)

    # Charger les données
    with open(preprocessed_path, 'r') as f:
        dataset = json.load(f)

    # Hyperparamètres
    n_iter = 10
    drop = 0.3

    # Démarrer une run MLflow
    with mlflow.start_run(run_name="spacy_ner_training"):
        # 1) Enregistrement des hyperparamètres
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("drop", drop)

        # 2) Entraînement du modèle
        nlp = train_spacy_ner(dataset, n_iter=n_iter, drop=drop, mlflow_run=True)

        # 3) Sauvegarde versionnée du modèle dans MLflow
        mlflow.spacy.log_model(spacy_model=nlp, artifact_path="spacy_model")

        # 4) Sauvegarde classique sur disque
        nlp.to_disk(model_out_dir)
        print(f"[train_model] Saved spaCy model => {model_out_dir}")

        # 5) Partage du chemin pour les prochaines tâches
        ti.xcom_push(key='model_dir', value=model_out_dir)

        # 6) Évaluation du modèle et journalisation des métriques dans MLflow
        precision, recall, f1 = evaluate_model(nlp, dataset)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


# --- Si exécuté en tant que script autonome ---
if __name__ == '__main__':
    # Example usage:
    in_file = os.path.join(DATA_DIR, "preprocessed_train.json")
    out_model_dir = os.path.join(MODELS_DIR, "my_spacy_model") 
    os.makedirs(out_model_dir, exist_ok=True)

    with open(in_file, 'r') as f:
        dataset = json.load(f)

    # Entraîner le modèle spaCy avec moins d'itérations pour un test rapide
    nlp = train_spacy_ner(dataset, n_iter=2, drop=0.3)
    nlp.to_disk(out_model_dir)
    print(f"[train_model main] Model saved => {out_model_dir}")
