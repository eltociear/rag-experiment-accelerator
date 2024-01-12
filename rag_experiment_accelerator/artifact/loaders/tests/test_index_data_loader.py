import json
import os
import shutil
import uuid

import pytest
from rag_experiment_accelerator.artifact.loaders.index_data_loader import (
    IndexDataLoader,
)
from rag_experiment_accelerator.artifact.models.index_data import IndexData
from rag_experiment_accelerator.artifact.writers.index_data_writer import (
    IndexDataWriter,
)
from rag_experiment_accelerator.artifact.loaders.tests.fixtures import temp_dir


def test_loads(temp_dir: str):
    test_data = [
        IndexData(name="index_name1"),
        IndexData(name="index_name2"),
    ]
    writer = IndexDataWriter(temp_dir)
    # save the data to a temp file so it can be loaded
    writer.save_all(test_data)

    # load the data
    loader = IndexDataLoader(writer.directory)
    loaded_data = loader.load_all()

    # assertions
    assert len(loaded_data) == len(test_data)
    assert [i.name for i in loaded_data] == [i.name for i in test_data]
