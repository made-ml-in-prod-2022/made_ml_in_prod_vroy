import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split(input_dir: Path, output_dir: Path, test_size: float):
    data = pd.read_csv(input_dir / "data.csv")
    target = pd.read_csv(input_dir / "target.csv")

    data['target'] = target.values
    train_df, test_df = train_test_split(data, test_size=test_size)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv")
    test_df.to_csv(output_dir / "test.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output-dir", "-o", type=str, required=True, 
        help="Path to splitted dataset")
    parser.add_argument("--test-size", "-s", type=float, default=0.2, help="Path to save dataset")
    args = parser.parse_args()
    split(Path(args.input_dir), Path(args.output_dir), args.test_size)


if __name__ == "__main__":
    main()