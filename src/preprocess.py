# scripts/preprocess.py
import json
import string
import os
from project_paths import DATA_DIR

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
    - Read fetched_data.json
    - Preprocess
    - Write preprocessed_data.json
    """
    ti = context['ti']
    fetched_data_path = ti.xcom_pull(key='fetched_data_path', task_ids='fetch_data')
    preprocessed_path = os.path.join(DATA_DIR, "preprocessed_data.json")
    os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)

    with open(fetched_data_path, 'r') as f:
        data = json.load(f)
    data = preprocess_data(data)
    with open(preprocessed_path, 'w') as out:
        json.dump(data, out, indent=2)
    print(f"[preprocess] Wrote preprocessed data => {preprocessed_path}")

    ti.xcom_push(key='preprocessed_data_path', value=preprocessed_path)

# --- If run as standalone script ---
if __name__ == '__main__':
    # Example usage:
    input_file = os.path.join(DATA_DIR, "dataturks.json")
    output_file = os.path.join(DATA_DIR, "preprocessed_data.json")
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    dataset = preprocess_data(dataset)
    with open(output_file, 'w') as out:
        json.dump(dataset, out, indent=2)
    print(f"[preprocess main] Output => {output_file}")
