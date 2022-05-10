
import logging
import os
import sys
import argparse

from ml_project.data import (
    download_data_from_gdrive,
    read_data,
    split_train_val_data
)
from ml_project.enities import (
    read_training_pipeline_params,
)
from ml_project.models import (
    create_model,
    train_model,
    save_model,
    evaluate_model,
    predict_model,
    save_metrics
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)

    return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    logger.info(f"Start train pipeline with params {training_pipeline_params}")

    downloading_params = training_pipeline_params.downloading_params
    os.makedirs(os.path.dirname(downloading_params.output_filepath), exist_ok=True)
    logger.info(f"Downloading train set from {downloading_params.train_set_url}...")
    download_data_from_gdrive(
        downloading_params.train_set_url,
        os.path.join(downloading_params.output_filepath, "train.csv")
    )

    data = read_data(training_pipeline_params.input_data_path)
    logger.debug(f"data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    val_target = val_df[training_pipeline_params.feature_params.target_col]
    train_target = train_df[training_pipeline_params.feature_params.target_col]
    train_df = train_df.drop(training_pipeline_params.feature_params.target_col, 1)
    val_df = val_df.drop(training_pipeline_params.feature_params.target_col, 1)
    logger.debug(f"train_df.shape is {train_df.shape}")
    logger.debug(f"val_df.shape is {val_df.shape}")

    logger.info("Training model...")
    model = create_model(training_pipeline_params.train_params)
    model = train_model(model, train_df, train_target)

    logger.info("Evaluate metrics...")
    predicts = predict_model(model, val_df)
    metrics = evaluate_model(predicts, val_target)
    logger.info(f"metrics is {metrics}")

    logger.info(f"Save metrics to the file {training_pipeline_params.metric_path}")
    save_metrics(metrics, training_pipeline_params.metric_path)

    logger.info(f"Save model to the file {training_pipeline_params.output_model_path}")
    save_model(model, training_pipeline_params.output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    args = parser.parse_args()

    train_pipeline(args.config)
