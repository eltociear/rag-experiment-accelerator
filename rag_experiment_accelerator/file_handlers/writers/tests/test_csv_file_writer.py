import csv
import os
import shutil
import uuid

import pytest

from rag_experiment_accelerator.file_handlers.writers.csv_file_writer import (
    CsvFileWriter,
)
import pandas as pd


@pytest.fixture()
def temp_dirname():
    dir = "/tmp/" + uuid.uuid4().__str__()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_write_raises_unsupported_data_type(temp_dirname: str):
    writer = CsvFileWriter()
    data = {"test_key_1": "test1", "test_key_2": "test2"}
    path = temp_dirname + "/test.jsonl"
    with pytest.raises(Exception) as e:
        writer.write(path, data)
    assert (
        str(e.value)
        == f"Unsupported data type {type(data)}. CSVFileWriter can only write pandas.DataFrame"
    )


def test_write(temp_dirname: str):
    writer = CsvFileWriter()
    data = [{"test_key_1": "test1", "test_key_2": "test2"}]
    df = pd.DataFrame(data)
    path = temp_dirname + "/test.csv"
    writer.write(path, df, index=False)
    assert os.path.exists(path)
    with open(path) as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                assert [r for r in row] == [
                    "test_key_1",
                    "test_key_2",
                ]
            if i == 1:
                assert [r for r in row] == ["test1", "test2"]
