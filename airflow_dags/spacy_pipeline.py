from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import your logic functions from separate scripts
from scripts.fetch_data import fetch_data_logic
from scripts.preprocess import preprocess_logic
from scripts.train_model import train_model_logic
from scripts.test_model import test_model_logic

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),  # your desired start date
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='spacy_model_pipeline',
    default_args=default_args,
    description='DAG that fetches data, preprocesses, trains, and tests a model',
    schedule_interval=None,  # or '0 12 * * *' for daily at noon
    catchup=False,
) as dag:

    # Task 1: Fetch data
    fetch_data_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data_logic  # calls the function in fetch_data.py
    )

    # Task 2: Preprocess
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_logic
    )

    # Task 3: Train model
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model_logic
    )

    # Task 4: Test model
    test_model_task = PythonOperator(
        task_id='test_model',
        python_callable=test_model_logic
    )

    # Set up dependencies (>> means "runs before")
    fetch_data_task >> preprocess_task >> train_model_task >> test_model_task
