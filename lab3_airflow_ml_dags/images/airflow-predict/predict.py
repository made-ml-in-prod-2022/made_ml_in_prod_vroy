import os
import pickle
import argparse
import pandas as pd
from pathlib import Path


def validate(input_dir_path: Path, model_path: Path, output_dir_path: Path):
    X = pd.read_csv(input_dir_path / "data.csv", index_col=0)
    
    print(X.columns)
    
    if "target" in X.columns:
        X = X.drop(columns="target").values

    print(X.columns)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    os.makedirs(output_dir_path, exist_ok=True)
    pred_df = pd.DataFrame({"prediction": y_pred})
    pred_df.to_csv(os.path.join(output_dir_path, "predictions.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Path to data directory")
    parser.add_argument("--model-path", "-m", type=str, required=True, help="Path to model")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Path to predictions dir")
    args = parser.parse_args()
    validate(Path(args.input_dir), Path(args.model_path), Path(args.output_dir))


if __name__ == "__main__":
    main()