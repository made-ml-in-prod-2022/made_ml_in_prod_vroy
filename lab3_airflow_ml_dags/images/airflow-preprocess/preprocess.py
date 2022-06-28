import os
import argparse
import pandas as pd
from pathlib import Path


def preprocess(input_dir: Path, output_dir: Path):
    data_df = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=0)
    target_df = pd.read_csv(os.path.join(input_dir, "target.csv"), index_col=0)
    
    # For example preproces here - delete some columns
    data_df = data_df.drop(columns=["alcalinity_of_ash", "nonflavanoid_phenols"])

    os.makedirs(output_dir, exist_ok=True)
    data_df.to_csv(os.path.join(output_dir, "data.csv"))
    target_df.to_csv(os.path.join(output_dir, "target.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="Path to processed dataset")
    args = parser.parse_args()
    preprocess(Path(args.input_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()