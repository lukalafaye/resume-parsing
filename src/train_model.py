# scripts/train_model.py
import json
import random
import os
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

from project_paths import DATA_DIR
from project_paths import MODELS_DIR

def train_spacy_ner(dataset, n_iter=15, drop=0.3):
    """
    Minimal spaCy training on a blank English pipeline, 
    using the approach from your final notebook.
    """
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Gather labels
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
    Airflow PythonOperator entry point for training.
    - Reads preprocessed_data.json
    - Trains the spaCy model
    - Saves model to disk
    """
    ti = context['ti']
    preprocessed_path = ti.xcom_pull(key='train_data_path', task_ids='preprocess_data')
    model_out_dir = os.path.join(MODELS_DIR, "my_spacy_model") 
    os.makedirs(model_out_dir, exist_ok=True)

    with open(preprocessed_path, 'r') as f:
        dataset = json.load(f)

    nlp = train_spacy_ner(dataset, n_iter=10, drop=0.3)

    nlp.to_disk(model_out_dir)
    print(f"[train_model] Saved spaCy model => {model_out_dir}")

    ti.xcom_push(key='model_dir', value=model_out_dir)

# --- If run as standalone script ---
if __name__ == '__main__':
    # Example usage:
    in_file = os.path.join(DATA_DIR, "preprocessed_train.json")
    out_model_dir = os.path.join(MODELS_DIR, "my_spacy_model") 
    os.makedirs(out_model_dir, exist_ok=True)

    with open(in_file, 'r') as f:
        dataset = json.load(f)

    nlp = train_spacy_ner(dataset, n_iter=2, drop=0.3)  # fewer iters for quick test
    nlp.to_disk(out_model_dir)
    print(f"[train_model main] Model saved => {out_model_dir}")
