import os

# Get the absolute path to the current file (this file)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root is the parent directory of src/
PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))

# Define paths to commonly used folders
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
AIRFLOW_DAGS_DIR = os.path.join(PROJECT_ROOT, 'airflow_dags')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def run():
    # Create any directories if they don't exist
    for path in [DATA_DIR, SRC_DIR, MODELS_DIR, RESULTS_DIR, AIRFLOW_DAGS_DIR]:
        os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    run()
