import json
import random
import os
import spacy
import mlflow
import mlflow.spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.metrics import classification_report
from collections import defaultdict

from project_paths import DATA_DIR
from project_paths import MODELS_DIR


def train_spacy_ner(dataset, n_iter=15, drop=0.3, run=None):
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
                aligned_ents = [
                    (start, end, label)
                    for (start, end, label) in ann["entities"]
                    if start < end
                ]
                example = Example.from_dict(doc, {"entities": aligned_ents})
                examples.append(example)
            nlp.update(examples, drop=drop, losses=losses, sgd=optimizer)

        print(f"Iter {itn} | Losses: {losses}")

        if run:
            for k, v in losses.items():
                mlflow.log_metric(f"loss_{k}", v, step=itn)

    return nlp


def evaluate_model(nlp, dataset, results_dir, mlflow_run=None):
    """
    Évalue le modèle sur le dataset fourni et sauvegarde les métriques dans MLflow et un fichier .txt
    """
    true_entities = []
    pred_entities = []

    for text, ann in dataset:
        doc = nlp(text)
        pred = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        true = [(start, end, label) for start, end, label in ann['entities']]

        true_entities.extend([ent[2] for ent in true])
        pred_entities.extend([ent[2] for ent in pred])

    report = classification_report(
        true_entities, pred_entities, output_dict=True, zero_division=0
    )
    report_text = classification_report(true_entities, pred_entities, zero_division=0)

    os.makedirs(results_dir, exist_ok=True)
    report_file = os.path.join(results_dir, "classification_report.txt")

    with open(report_file, "w") as f:
        f.write(report_text)
    print(f"[evaluate_model] Rapport écrit dans => {report_file}")

    if mlflow_run:
        mlflow.log_artifact(report_file, artifact_path="results")

        # Log global averages
        for avg_type in ['micro avg', 'macro avg', 'weighted avg']:
            for metric in ['precision', 'recall', 'f1-score', 'support']:
                val = report.get(avg_type, {}).get(metric)
                if val is not None:
                    mlflow.log_metric(f"{avg_type.replace(' ', '_')}_{metric}", val)

        # Log par label
        for label, metrics in report.items():
            if label in ['micro avg', 'macro avg', 'weighted avg', 'accuracy']:
                continue
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", metric_value)


def train_model_logic(**context):
    ti = context['ti']
    preprocessed_path = ti.xcom_pull(key='train_data_path', task_ids='preprocess_data')

    model_out_dir = os.path.join(MODELS_DIR, "my_spacy_model")
    results_dir = os.path.join("results")
    os.makedirs(model_out_dir, exist_ok=True)

    with open(preprocessed_path, 'r') as f:
        dataset = json.load(f)

    with mlflow.start_run(run_name="spacy_ner_training") as run:
        mlflow.set_tags({
            "author": "ton_nom",
            "model_type": "spaCy_NER",
            "source": "Airflow"
        })

        mlflow.log_param("n_iter", 10)
        mlflow.log_param("dropout", 0.3)
        mlflow.log_param("dataset_size", len(dataset))
        mlflow.log_artifact(preprocessed_path, artifact_path="data")

        nlp = train_spacy_ner(dataset, n_iter=10, drop=0.3, run=run)

        nlp.to_disk(model_out_dir)
        print(f"[train_model] Saved spaCy model => {model_out_dir}")

        example_texts = [text for text, _ in dataset[:5]]
        input_df = pd.DataFrame({"text": example_texts})
        signature = infer_signature(input_df)

        mlflow.spacy.log_model(
            spacy_model=nlp,
            artifact_path="model",
            signature=signature,
            input_example=input_df
        )

        evaluate_model(nlp, dataset, results_dir, mlflow_run=run)
        ti.xcom_push(key='model_dir', value=model_out_dir)


# --- If run as standalone script ---
if __name__ == '__main__':
    in_file = os.path.join(DATA_DIR, "preprocessed_train.json")
    out_model_dir = os.path.join(MODELS_DIR, "my_spacy_model")
    results_dir = os.path.join("results")
    os.makedirs(out_model_dir, exist_ok=True)

    with open(in_file, 'r') as f:
        dataset = json.load(f)

    with mlflow.start_run(run_name="spacy_ner_manual_run") as run:
        mlflow.set_tags({
            "author": "Robert",
            "model_type": "spaCy_NER",
            "source": "manual_script"
        })

        mlflow.log_param("n_iter", 10)
        mlflow.log_param("dropout", 0.3)
        mlflow.log_param("dataset_size", len(dataset))
        mlflow.log_artifact(in_file, artifact_path="data")

        nlp = train_spacy_ner(dataset, n_iter=10, drop=0.3, run=run)
        nlp.to_disk(out_model_dir)

        example_texts = [text for text, _ in dataset[:5]]
        input_df = pd.DataFrame({"text": example_texts})
        signature = infer_signature(input_df)

        mlflow.spacy.log_model(
            spacy_model=nlp,
            artifact_path="model",
            signature=signature,
            input_example=input_df
        )

        #evaluate_model(nlp, dataset, results_dir, mlflow_run=run)
        print(f"[train_model main] Model saved => {out_model_dir}")