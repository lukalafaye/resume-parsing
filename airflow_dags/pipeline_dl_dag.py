# airflow-dags/pipeline_dag.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import datetime

import os
import sys

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(SRC_PATH)

# Your modules
from fetch_data import fetch_data_logic
from preprocess import preprocess_logic
from train_model import train_model_logic
from test_model import test_model_logic


with DAG(
    dag_id="pipeline_dl",
    start_date=datetime.datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["dl", "example"]
):
    # 1) Fetch Data
    fetch_data_task = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data_logic,
        provide_context=True
    )

    # 2) Preprocess Data
    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_logic,
        provide_context=True
    )

    # 3) Train Model
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model_logic,
        provide_context=True
    )

    # 4) Test/Evaluate Model
    test_model_task = PythonOperator(
        task_id="test_model",
        python_callable=test_model_logic,
        provide_context=True
    )

    fetch_data_task >> preprocess_task >> train_model_task >> test_model_task

# --- If run as standalone script (uncommon) ---
if __name__ == "__main__":
    print("This DAG is meant to be used by Airflow, not run directly.")
    print("Use 'airflow tasks test my_spacy_ner_pipeline fetch_data [date]' or trigger in the Airflow UI.")