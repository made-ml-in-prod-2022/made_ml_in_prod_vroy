import os
from datetime import timedelta

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.python import PythonSensor
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount


from constants import DEFAULT_ARGS, HOST_DATA_DIR, RAW_DATA_DIR, PREPROCESSED_DATA_DIR, PREDICTION_DIR, MODEL_DIR, MODEL_PATH, START_DATE


def _wait_for_data(path):
    return os.path.exists(path)


with DAG(
        "prediction",
        default_args=DEFAULT_ARGS,
        schedule_interval="@daily",
        start_date=START_DATE,
) as dag:
    wait_data = PythonSensor(
        task_id="wait-for-data",
        python_callable=_wait_for_data,
        op_args=[os.path.join(RAW_DATA_DIR, "data.csv")],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    wait_target = PythonSensor(
        task_id="wait-for-target",
        python_callable=_wait_for_data,
        op_args=[os.path.join(RAW_DATA_DIR, "target.csv")],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    wait_model = PythonSensor(
        task_id="wait-for-mode",
        python_callable=_wait_for_data,
        op_args=[MODEL_PATH],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke",
    )

    preprocess = DockerOperator(
        task_id="preprocess-data",
        image="airflow-preprocess",
        command=f" --input-dir {RAW_DATA_DIR}"
                f" --output-dir {PREPROCESSED_DATA_DIR}",
        network_mode="bridge",
        do_xcom_push=False,
        mounts=[Mount(source=HOST_DATA_DIR, target="/data", type='bind')]
    )

    predict = DockerOperator(
        task_id="predict",
        image="airflow-predict",
        command=f" --input-dir {PREPROCESSED_DATA_DIR}"
                f" --model-path {MODEL_PATH}"
                f" --output-dir {PREDICTION_DIR}",
        network_mode="bridge",
        do_xcom_push=False,
        mounts=[Mount(source=HOST_DATA_DIR, target="/data", type='bind')]
    )

    [wait_data, wait_target, wait_model] >> preprocess >> predict

