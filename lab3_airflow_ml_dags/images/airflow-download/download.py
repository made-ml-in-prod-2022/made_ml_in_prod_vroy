import argparse
from pathlib import Path

from sklearn.datasets import load_wine


def download(output_dir):
    data, target = load_wine(return_X_y=True, as_frame=True)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    data.to_csv(output_dir_path / "data.csv", index=False)
    target.to_csv(output_dir_path / "target.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "-o", type=str, required=True, 
        help="Path to save dataset")
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()