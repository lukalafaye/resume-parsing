# scripts/train_model.py
import json
import random
import os
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

import mlflow
import mlflow.spacy
from sklearn.metrics import precision_score, recall_score, f1_score

from project_paths import DATA_DIR
from project_paths import MODELS_DIR

def evaluate_model(nlp, dataset):
    """
    Évalue le modèle spaCy en calculant précision, rappel, F1-score.
    """
    y_true = []
    y_pred = []

    for text, ann in dataset:
        doc = nlp(text)
        true_entities = [label for start, end, label in ann["entities"]]
        predicted_entities = [ent.label_ for ent in doc.ents]

        y_true.extend(true_entities)
        y_pred.extend(predicted_entities)

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return precision, recall, f1

def train_spacy_ner(dataset, n_iter=15, drop=0.3):
    """
    Entraîne un modèle spaCy NER à partir d’un dataset formaté.
    """
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for text, ann in dataset:
        for start, end, label in ann.get("entities", []):
            ner.add_label(label)

    optimizer = nlp.initialize()

    for itn in range(n_iter):
        random.shuffle(dataset)
        losses = {}
        batches = minibatch(dataset, size=compounding(4.0, 32.0, 1.001))

        for batch in batches:
            examples = []
            for text, ann in batch:
                doc = nlp.make_doc(text)
                aligned_ents = []
                for (start, end, label) in ann["entities"]:
                    if start < end:
                        aligned_ents.append([start, end, label])
                example = Example.from_dict(doc, {"entities": aligned_ents})
                examples.append(example)
            nlp.update(examples, drop=drop, losses=losses, sgd=optimizer)

        print(f"Iter {itn} | Losses: {losses}")
    return nlp

def train_model_logic(**context):
    """
    Point d’entrée Airflow pour entraîner et logger un modèle spaCy avec MLflow.
    """
    ti = context['ti']
    preprocessed_path = ti.xcom_pull(key='train_data_path', task_ids='preprocess_data')
    model_out_dir = os.path.join(MODELS_DIR, "my_spacy_model") 
    os.makedirs(model_out_dir, exist_ok=True)

    with open(preprocessed_path, 'r') as f:
        dataset = json.load(f)

    n_iter = 10
    drop = 0.3

    with mlflow.start_run(run_name="spacy_ner_training"):
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("drop", drop)

        nlp = train_spacy_ner(dataset, n_iter=n_iter, drop=drop)

        mlflow.spacy.log_model(spacy_model=nlp, artifact_path="spacy_model")

        nlp.to_disk(model_out_dir)
        print(f"[train_model] Saved spaCy model => {model_out_dir}")

        ti.xcom_push(key='model_dir', value=model_out_dir)

        precision, recall, f1 = evaluate_model(nlp, dataset)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

# --- Exécution standalone ---
if __name__ == '__main__':
    in_file = os.path.join(DATA_DIR, "preprocessed_train.json")
    out_model_dir = os.path.join(MODELS_DIR, "my_spacy_model") 
    os.makedirs(out_model_dir, exist_ok=True)

    with open(in_file, 'r') as f:
        dataset = json.load(f)

    nlp = train_spacy_ner(dataset, n_iter=2, drop=0.3)
    nlp.to_disk(out_model_dir)
    print(f"[train_model main] Model saved => {out_model_dir}")
