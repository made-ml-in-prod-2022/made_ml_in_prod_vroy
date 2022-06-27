import os
import json
import pickle
import argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score


def validate(input_path: Path, model_path: Path, metric_path: Path):
    X_test = pd.read_csv(input_path, index_col=0)
    
    y_test = X_test['target'].values
    X_test = X_test.drop(columns="target").values

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy_score": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
    }

    os.makedirs(metric_path.parent, exist_ok=True)
    with open(metric_path, "w") as f:
        json.dump(metrics , f) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", "-i", type=str, required=True, help="Path to .csv data")
    parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to model")
    parser.add_argument("--metric-path", "-o", type=str, required=True, help="Output path to metric.json")
    args = parser.parse_args()
    validate(Path(args.input_path), Path(args.model_path), Path(args.metric_path))


if __name__ == "__main__":
    main()