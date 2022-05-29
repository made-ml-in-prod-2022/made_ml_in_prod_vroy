import json
import pickle
from typing import Dict, Union, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from ml_project.enities.train_params import TrainParams

SklearnClassificationModel = Union[KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]


def create_model(train_params: TrainParams) -> SklearnClassificationModel:
    model_params = train_params.model_params
    if train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=model_params.n_neighbors,
        )
    elif train_params.model_type == "DecisionTreeClassifier":
        model = DecisionTreeClassifier(
            criterion=model_params.criterion,
            max_depth=model_params.max_depth,
            min_samples_split=model_params.min_samples_split,
            random_state=model_params.random_state
        )
    elif train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier()
    else:
        raise NotImplementedError()
    return model


def train_model(
        model: SklearnClassificationModel,
        features: pd.DataFrame, target: pd.Series,
) -> SklearnClassificationModel:
    model.fit(features, target)
    return model


def predict_model(model: Union[Pipeline, SklearnClassificationModel], features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "acc": accuracy_score(target, predicts),
        "r2_score": r2_score(target, predicts),
        "rmse": mean_squared_error(target, predicts, squared=False),
        "mae": mean_absolute_error(target, predicts),
    }


def create_inference_pipeline(
    model: SklearnClassificationModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def save_metrics(metrics: dict, metric_path: str):
    with open(metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)


def save_predictions(predictions: Union[List[int], np.ndarray], predictions_path: str):
    with open(predictions_path, "w") as f:
        f.write("\n".join(str(item) for item in predictions))


def save_model(model: object, output: str):
    with open(output, "wb") as f:
        pickle.dump(model, f)


def load_model(path_to_model: str) -> SklearnClassificationModel:
    with open(path_to_model, "rb") as f:
        model = pickle.load(f)
    return model
