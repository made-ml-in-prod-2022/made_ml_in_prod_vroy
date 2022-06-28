import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from constants import DEFAULT_ARGS, HOST_DATA_DIR, RAW_DATA_DIR, START_DATE


with DAG(
        "download_data",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=START_DATE,
) as dag:
    download = DockerOperator(
        image="airflow-download",
        command=f"-o {RAW_DATA_DIR}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        # do_xcom_push=True,
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_DATA_DIR, target="/data", type='bind')]
    )
