import os
import datetime

from airflow.models import Variable
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}
START_DATE = days_ago(8)

HOST_DATA_DIR = Variable.get("host_data_dir")
RAW_DATA_DIR = "/data/raw/{{ ds }}"