import os
import datetime

from airflow.models import Variable
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}
START_DATE = days_ago(8)

HOST_DATA_DIR = Variable.get("host_data_dir")
RAW_DATA_DIR = "/data/raw/{{ ds }}"
SPLIT_DATA_DIR = "/data/split/{{ ds }}"
PREPROCESSED_DATA_DIR = "/data/processed/{{ ds }}"
MODEL_DIR = "/data/models/{{ ds }}"