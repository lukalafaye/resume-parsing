# scripts/preprocess.py
import json
import string
import os
from project_paths import DATA_DIR
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    """
    Main pipeline of cleaning steps:
      1) fix_reversed_offsets
      2) remove_trailing_spaces
      3) trim_entity_boundaries
      4) expand_partial_word_slices
      5) split_or_skip_large_spans
      6) remove_overlapping_entities
    """
    data = fix_reversed_offsets(data)
    data = remove_trailing_spaces(data)
    data = trim_entity_boundaries(data)
    data = expand_partial_word_slices(data)
    data = split_or_skip_large_spans(data)
    data = remove_overlapping_entities(data)
    return data

#--- Sub-steps from your final notebook pipeline ---
def fix_reversed_offsets(data):
    for entry in data:
        text, ann = entry
        for ent in ann['entities']:
            start, end, label = ent
            if start > end:
                ent[0], ent[1] = end, start
    return data

def remove_trailing_spaces(data):
    for i, entry in enumerate(data):
        text, ann = entry
        new_text = text.rstrip()
        if new_text != text:
            for ent in ann['entities']:
                if ent[1] > len(new_text):
                    ent[1] = len(new_text)
            entry[0] = new_text
    return data

def trim_entity_boundaries(data):
    for i, (text, ann) in enumerate(data):
        new_entities = []
        for (start, end, label) in ann["entities"]:
            while start < end and text[start] in string.whitespace + string.punctuation:
                start += 1
            while end > start and text[end-1] in string.whitespace + string.punctuation:
                end -= 1
            if start < end:
                new_entities.append([start, end, label])
        ann["entities"] = new_entities
    return data

def expand_partial_word_slices(data):
    for i, (text, ann) in enumerate(data):
        new_entities = []
        for start, end, label in ann["entities"]:
            while start > 0 and not text[start-1].isspace():
                start -= 1
            while end < len(text) and not text[end].isspace():
                end += 1
            new_entities.append([start, end, label])
        ann["entities"] = new_entities
    return data

def split_or_skip_large_spans(data, label="Skills", max_span=100):
    for i, (text, ann) in enumerate(data):
        new_entities = []
        for (start, end, lbl) in ann["entities"]:
            length = end - start
            if lbl == label and length > max_span:
                chunk = text[start:end]
                segments = [s.strip() for s in chunk.split(',')]
                offset = start
                for seg in segments:
                    if not seg:
                        continue
                    seg_start = text.find(seg, offset)
                    if seg_start == -1:
                        continue
                    seg_end = seg_start + len(seg)
                    offset = seg_end
                    if (seg_end - seg_start) <= max_span:
                        new_entities.append([seg_start, seg_end, lbl])
            else:
                new_entities.append([start, end, lbl])
        ann["entities"] = new_entities
    return data

def remove_overlapping_entities(data):
    def overlaps(s1, e1, s2, e2):
        return not (e1 <= s2 or e2 <= s1)
    for i, (text, ann) in enumerate(data):
        ents = ann["entities"]
        ents_sorted = sorted(ents, key=lambda e: (e[1] - e[0]))  # short first
        final = []
        for start, end, lbl in ents_sorted:
            if not any(overlaps(start, end, f[0], f[1]) for f in final):
                final.append([start, end, lbl])
        ann["entities"] = final
    return data

def preprocess_logic(**context):
    """
    Airflow PythonOperator entry point:
      • read fetched_data.json
      • preprocess
      • split train / test
      • write preprocessed_train.json , preprocessed_test.json
    """
    ti = context["ti"]

    # --- 1. read raw data ----------------------------------------------------
    fetched_data_path = ti.xcom_pull(key="fetched_data_path",
                                     task_ids="fetch_data")
    with open(fetched_data_path, "r") as f:
        data = json.load(f)

    # --- 2. clean ------------------------------------------------------------
    data = preprocess_data(data)

    # --- 3. split ------------------------------------------------------------
    train, test = train_test_split(
        data,
        test_size=0.2,      # 80 % / 20 % by default
        random_state=42,    # for reproducibility
        shuffle=True,
    )

    # --- 4. save -------------------------------------------------------------
    train_path = os.path.join(DATA_DIR, "preprocessed_train.json")
    test_path  = os.path.join(DATA_DIR, "preprocessed_test.json")
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test, f, indent=2)

    print(f"[preprocess] → {train_path}  ({len(train)} samples)")
    print(f"[preprocess] → {test_path}   ({len(test)} samples)")

    # --- 5. push paths to XCom ----------------------------------------------
    ti.xcom_push(key="train_data_path", value=train_path)
    ti.xcom_push(key="test_data_path",  value=test_path)

# --- If run as standalone script ---
# --- If run as standalone script -------------------------------------------
if __name__ == "__main__":
    """
    Example:
        python src/preprocess.py  --input data/dataturks.json  --ratio 0.25
    """
    import argparse
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser(description="Pre‑process & split dataset")
    parser.add_argument(
        "--input",
        default=os.path.join(DATA_DIR, "dataturks.json"),
        help="Path to raw DataTurks‑format JSON file",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.20,
        help="Test split ratio (e.g. 0.2 = 20 %)",
    )
    args = parser.parse_args()

    # 1) load
    with open(args.input, "r") as f:
        dataset = json.load(f)

    # 2) clean
    dataset = preprocess_data(dataset)

    # 3) split
    train_set, test_set = train_test_split(
        dataset, test_size=args.ratio, random_state=42, shuffle=True
    )

    # 4) save
    train_path = os.path.join(DATA_DIR, "preprocessed_train.json")
    test_path  = os.path.join(DATA_DIR, "preprocessed_test.json")

    with open(train_path, "w") as f:
        json.dump(train_set, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_set, f, indent=2)

    print(f"[preprocess main]  → {train_path}  ({len(train_set)} samples)")
    print(f"[preprocess main]  → {test_path}   ({len(test_set)} samples)")
