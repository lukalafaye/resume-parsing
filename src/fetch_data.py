# scripts/fetch_data.py
from project_paths import DATA_DIR
import json
import os

def labelstudio_to_dataturks_format(ls_file, output_file):
    """
    Loads a Label Studio JSON and converts it to DataTurks-like format:
    [
      [text, {"entities": [ [start, end, label], ... ]}],
      ...
    ]
    Writes the result to `output_file` in JSON.
    """
    with open(ls_file, 'r') as f:
        data = json.load(f)

    converted = []
    for task in data:
        text = task['data']['text']
        entities = []
        if 'annotations' in task and len(task['annotations']) > 0:
            results = task['annotations'][0].get('result', [])
            for r in results:
                if r['type'] == 'labels':
                    start = r['value']['start']
                    end = r['value']['end']
                    label = r['value']['labels'][0]
                    entities.append([start, end, label])
        converted.append([text, {'entities': entities}])

    with open(output_file, 'w') as out:
        json.dump(converted, out, indent=2)
    print(f"[fetch_data] Wrote {len(converted)} samples to {output_file}")


def fetch_data_logic(**context):
    """
    Airflow PythonOperator entry point.
    1) Convert Label Studio JSON to DataTurks-like format
    2) Optionally push path to XCom for the next task
    """
    input_file = os.path.join(DATA_DIR, "dataset.json")
    output_file = os.path.join(DATA_DIR, "dataturks.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    labelstudio_to_dataturks_format(input_file, output_file)

    # You can push the output path via XCom
    ti = context['ti']
    ti.xcom_push(key='fetched_data_path', value=output_file)

if __name__ == "__main__":
    input_file = os.path.join(DATA_DIR, "dataset.json")
    output_file = os.path.join(DATA_DIR, "dataturks.json")
    labelstudio_to_dataturks_format(input_file, output_file)