import os
import shutil
import uuid

import pytest

from rag_experiment_accelerator.file_handlers.writers.jsonl_file_writer import (
    JsonlFileWriter,
)


@pytest.fixture()
def temp_dirname():
    dir = "/tmp/" + uuid.uuid4().__str__()
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)


def test_write(temp_dirname: str):
    writer = JsonlFileWriter()
    data = {"test": "test"}
    path = temp_dirname + "/test.jsonl"
    writer.write(path, data)
    with open(path) as file:
        assert file.readline() == '{"test": "test"}\n'
