import argparse
import numpy as np
import pandas as pd
import requests


def main(dataset_path: str, url: str):
    data_df = pd.read_csv(dataset_path)
    print(data_df.head())
    request_features = list(data_df.columns)
    print(request_features)
    request_data = []
    for _, row in data_df.iterrows():
        request_data_row = [
            x.item() if isinstance(x, np.generic) else x for x in row.tolist()
        ]
        request_data.append(request_data_row)

    print(request_data[0])
    request_json = {"data": request_data, "features": request_features}
    response = requests.get(
        f"{url}/predict/",
        json=request_json,
    )
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", "-i", required=True, help="Path to dataset.csv file")
    parser.add_argument("--url", "-u", required=False, default="http://localhost:8000",
                        help="Url of RestAPI service")
    args = parser.parse_args()

    main(args.data, args.url)
