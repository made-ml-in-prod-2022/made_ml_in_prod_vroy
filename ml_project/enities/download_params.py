from dataclasses import dataclass


@dataclass()
class DownloadParams:
    train_set_url: str
    test_set_url: str
    output_filepath: str
