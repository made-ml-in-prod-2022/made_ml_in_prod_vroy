import pytest
import os

from ml_project.data.make_dataset import download_data_from_gdrive


DEFAULT_URL = "https://drive.google.com/file/d/1oGsm_g0qKRduKanE3C1VCML0EM4CS3pm/view?usp=sharing"
DEFAULT_DOWNLADED_FILENAME = "train.csv"


def test_can_download_file_from_gdrive(tmpdir):
    output_filepath = os.path.join(tmpdir, DEFAULT_DOWNLADED_FILENAME)
    download_data_from_gdrive(DEFAULT_URL, output_filepath)

    assert os.path.isfile(output_filepath)
