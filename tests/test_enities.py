import yaml
import pytest

from marshmallow_dataclass import class_schema
from marshmallow.exceptions import ValidationError
from typing import Union, TextIO

from ml_project.enities import (
    DownloadParams, SplittingParams, FeatureParams, TrainParams, ModelParams,
    read_training_pipeline_params
)


DEFAULT_TEST_CONFIG_PATH = "../configs/train_config_knn.yaml"


def read_schema(input_stream: Union[str, TextIO], schema):
    return schema().load(yaml.safe_load(input_stream))


def test_can_read_downloading_params():
    test_string = """
        url: "https://drive.google.com"
        output_filepath: "data/"
    """
    try:
        params = read_schema(test_string, class_schema(DownloadParams))
    except ValidationError:
        raise pytest.fail("Can't read DownloadParams")

    assert params.url == "https://drive.google.com"
    assert params.output_filepath == "data/"


def test_can_read_splitting_params():
    test_string = """
          val_size: 0.2
          random_state: 3
    """
    try:
        params = read_schema(test_string, class_schema(SplittingParams))
    except ValidationError:
        raise pytest.fail("Can't read SplittingParams")

    assert params.val_size == 0.2
    assert params.random_state == 3


def test_can_read_train_params():
    test_string = """
          model_type: "KNeighborsClassifier"
          model_params:
            n_neighbors: 22
    """
    try:
        params = read_schema(test_string, class_schema(TrainParams))
    except ValidationError:
        raise pytest.fail("Can't read TrainParams")

    assert params.model_type == "KNeighborsClassifier"
    assert type(params.model_params) == ModelParams
    assert params.model_params.n_neighbors == 22
    assert params.model_params.random_state == 17


def test_can_read_feature_params():
    test_string = """
        categorical_features:
          - "sex"
          - "religion"
        numerical_features:
          - "age"
          - "weight"
        features_to_drop:
          - "drop"
        target_col: "target"
    """
    try:
        params = read_schema(test_string, class_schema(FeatureParams))
    except ValidationError:
        raise pytest.fail("Can't read FeatureParams")

    assert params.categorical_features == ["sex", "religion"]
    assert params.numerical_features == ["age", "weight"]
    assert params.features_to_drop == ["drop"]
    assert params.target_col == "target"


def test_can_read_train_pipeline_params():
    try:
        params = read_training_pipeline_params(DEFAULT_TEST_CONFIG_PATH)
    except ValidationError:
        raise pytest.fail("Can't read TrainParams")

    assert params.input_data_path == "data/heart_cleveland_upload.csv"
    assert params.output_model_path == "models/model.pkl"
    assert type(params.downloading_params) == DownloadParams
    assert type(params.splitting_params) == SplittingParams
    assert type(params.train_params) == TrainParams
    assert type(params.feature_params) == FeatureParams
