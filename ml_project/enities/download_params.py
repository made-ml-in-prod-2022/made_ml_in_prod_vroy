from dataclasses import dataclass


@dataclass()
class DownloadParams:
    url: str
    output_filepath: str
