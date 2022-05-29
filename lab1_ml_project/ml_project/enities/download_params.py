from dataclasses import dataclass


@dataclass()
class DownloadParams:
    train_set_url: str
    test_set_url: str
    train_set_path: str
    test_set_path: str
