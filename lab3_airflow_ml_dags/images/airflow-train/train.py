import os
import argparse
import pickle
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression


def train(input_path: Path, model_path: Path):
    X_train = pd.read_csv(input_path, index_col=0)
    
    y_train = X_train['target'].values
    X_train = X_train.drop(columns="target").values

    model = LogisticRegression(max_iter=10000, multi_class="multinomial", solver="saga", penalty="l1")
    model.fit(X_train, y_train)

    os.makedirs(model_path.parent, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", "-i", type=str, required=True, help="Path to .csv data")
    parser.add_argument("--model-path", "-o", type=str, required=True, help="Output path to model")
    args = parser.parse_args()
    train(Path(args.input_path), Path(args.model_path))


if __name__ == "__main__":
    main()