import os
import yaml
import pytest

from marshmallow_dataclass import class_schema
from marshmallow.exceptions import ValidationError
from typing import Union, TextIO

from ml_project.enities import (
    DownloadParams, SplittingParams, FeatureParams, TrainParams, ModelParams,
    read_training_pipeline_params
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_CONFIG_PATH = os.path.join(CURRENT_DIR, "test_data/test_config.yaml")


def read_schema(input_stream: Union[str, TextIO], schema):
    return schema().load(yaml.safe_load(input_stream))


def test_can_read_downloading_params():
    test_string = """
        train_set_url: "https://drive.google.com/url1"
        test_set_url: "https://drive.google.com/url2"
        train_set_path: "data/train.csv"
        test_set_path: "data/test.csv"
    """
    try:
        params = read_schema(test_string, class_schema(DownloadParams))
    except ValidationError:
        raise pytest.fail("Can't read DownloadParams")

    assert params.train_set_url == "https://drive.google.com/url1"
    assert params.test_set_url == "https://drive.google.com/url2"
    assert params.train_set_path == "data/train.csv"
    assert params.test_set_path == "data/test.csv"


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


def test_can_read_knn_train_params():
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


def test_can_read_decision_tree_train_params():
    test_string = """
          model_type: "DecisionTreeClassifier"
          model_params:
            criterion: "entropy"
            max_depth: 7
            min_samples_split: 5
            random_state: 17
    """
    try:
        params = read_schema(test_string, class_schema(TrainParams))
    except ValidationError:
        raise pytest.fail("Can't read TrainParams")

    assert params.model_type == "DecisionTreeClassifier"
    assert type(params.model_params) == ModelParams
    assert params.model_params.criterion == "entropy"
    assert params.model_params.max_depth == 7
    assert params.model_params.min_samples_split == 5
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

    assert params.input_data_path == "data/dataset.csv"
    assert params.output_model_path == "models/model.pkl"
    assert params.input_test_path == "data/test.csv"
    assert params.output_predictions_path == "models/predictions.txt"
    assert type(params.downloading_params) == DownloadParams
    assert type(params.splitting_params) == SplittingParams
    assert type(params.train_params) == TrainParams
    assert type(params.feature_params) == FeatureParams
