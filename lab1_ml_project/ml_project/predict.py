
import logging
import os
import sys
import argparse

from ml_project.data import (
    download_data_from_gdrive,
    read_data,
)
from ml_project.enities import (
    read_training_pipeline_params,
)
from ml_project.models import (
    load_model,
    predict_model,
    save_predictions
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)

    downloading_params = training_pipeline_params.downloading_params
    os.makedirs(os.path.dirname(downloading_params.test_set_path), exist_ok=True)
    logger.info(f"Downloading test set from {downloading_params.test_set_url}...")
    download_data_from_gdrive(
        downloading_params.test_set_url,
        downloading_params.test_set_path
    )

    test_data = read_data(training_pipeline_params.input_test_path)
    logger.debug(f"test_data.shape is {test_data.shape}")

    logger.info(f"Loading existing model from file {training_pipeline_params.output_model_path}")
    model = None
    if os.path.isfile(training_pipeline_params.output_model_path):
        model = load_model(training_pipeline_params.output_model_path)
    else:
        logger.error(f"Can't load model - file is not exist!")
        exit(-1)

    logger.info("Predict...")
    predicts = predict_model(model, test_data)

    logger.info(f"Save predictions to the file {training_pipeline_params.output_predictions_path}")
    save_predictions(predicts, training_pipeline_params.output_predictions_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    args = parser.parse_args()

    predict_pipeline(args.config)
