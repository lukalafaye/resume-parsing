import json
import spacy
from spacy.training import offsets_to_biluo_tags
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from itertools import chain
import os
import mlflow
import mlflow.spacy

from project_paths import DATA_DIR, RESULTS_DIR, MODELS_DIR

def evaluate_ner_model(nlp, data):
    y_true = []
    y_pred = []
    for text, ann in data:
        gold_ents = ann.get("entities", [])
        doc = nlp.make_doc(text)
        gold_tags = offsets_to_biluo_tags(doc, gold_ents)

        pred_doc = nlp(text)
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents]
        pred_tags = offsets_to_biluo_tags(doc, pred_entities)

        y_true.append(gold_tags)
        y_pred.append(pred_tags)
    return ner_report(y_true, y_pred)

def ner_report(y_true, y_pred):
    y_true_flat = list(chain.from_iterable(y_true))
    y_pred_flat = list(chain.from_iterable(y_pred))

    overall_acc = accuracy_score(y_true_flat, y_pred_flat)

    non_o_indices = [i for i, tag in enumerate(y_true_flat) if tag != 'O']
    if not non_o_indices:
        return ("No entities found in dataset!", overall_acc, None)

    y_true_filtered = [y_true_flat[i] for i in non_o_indices]
    y_pred_filtered = [y_pred_flat[i] for i in non_o_indices]

    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(y_true_filtered)
    y_pred_combined = lb.transform(y_pred_filtered)

    tagset = sorted(lb.classes_, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    report = classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
        zero_division=0
    )
    entity_acc = accuracy_score(y_true_combined, y_pred_combined)
    return report, overall_acc, entity_acc

def test_model_logic(**context):
    """
    Airflow PythonOperator entry point for testing.
    - Loads the trained model
    - Uses preprocessed_data.json as test
    - Prints metrics
    """
    ti = context['ti']
    model_dir = ti.xcom_pull(key='model_dir', task_ids='train_model')
    preprocessed_path = ti.xcom_pull(key='test_data_path', task_ids='preprocess_data')

    with open(preprocessed_path, 'r') as f:
        dataset = json.load(f)

    nlp = spacy.load(model_dir)

    report, overall_acc, entity_acc = evaluate_ner_model(nlp, dataset)
    print("=== EVALUATION RESULTS ===")
    if isinstance(report, str):
        print(report)
    else:
        print(report)
        print(f"Overall Accuracy (with 'O'): {overall_acc:.4f}")
        if entity_acc is not None:
            print(f"Entity-only Accuracy: {entity_acc:.4f}")

    # Démarrer une session MLflow
    with mlflow.start_run():
        # Sauvegarder le modèle
        mlflow.spacy.log_model(nlp, "model")

        # Log des métriques dans MLflow
        mlflow.log_metric("overall_accuracy", overall_acc)
        if entity_acc is not None:
            mlflow.log_metric("entity_accuracy", entity_acc)

        # Log des résultats détaillés du rapport de classification
        for line in report.splitlines():
            mlflow.log_param("classification_report_line", line)

    # Sauvegarder les résultats dans un fichier
    results_path = os.path.join(RESULTS_DIR, "test_results.txt")
    with open(results_path, 'w') as out:
        out.write("=== EVALUATION RESULTS ===\n")
        if isinstance(report, str):
            out.write(report + "\n")
        else:
            out.write(report + "\n")
            out.write(f"Overall Accuracy: {overall_acc:.4f}\n")
            if entity_acc:
                out.write(f"Entity-only Accuracy: {entity_acc:.4f}\n")
    print(f"[test_model] Wrote results to => {results_path}")

# --- If run as standalone script ---
if __name__ == '__main__':

    in_file = os.path.join(DATA_DIR, "preprocessed_data.json")
    model_dir = os.path.join(MODELS_DIR, "my_spacy_model")

    with open(in_file, 'r') as f:
        dataset = json.load(f)

    nlp = spacy.load(model_dir)
    report, overall_acc, entity_acc = evaluate_ner_model(nlp, dataset)
    print("=== EVALUATION RESULTS (standalone) ===")
    if isinstance(report, str):
        print(report)
    else:
        print(report)
        print(f"Overall Accuracy (with 'O'): {overall_acc:.4f}")
        if entity_acc:
            print(f"Entity-only Accuracy: {entity_acc:.4f}")
