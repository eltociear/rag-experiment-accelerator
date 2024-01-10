import json
import os
import pathlib
import shutil
import tempfile

import pytest
from rag_experiment_accelerator.artifact.managers.index_manager import IndexDataManager
from rag_experiment_accelerator.artifact.models.index import Index


@pytest.fixture()
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_save(temp_dir: str):
    manager = IndexDataManager(temp_dir)
    index_1 = Index("prefix", 1, 1, "embedding_model_name", 1, 1)
    index_2 = Index("prefix", 2, 2, "embedding_model_name", 2, 2)
    indexes = [index_1, index_2]

    manager.save(indexes)
    assert pathlib.Path(manager.output_filepath).exists()
    with open(manager.output_filepath, "r") as file:
        loaded_index_data = json.loads(file.read())
        assert type(loaded_index_data["indexes"]) == list
        for i, index_name in enumerate(loaded_index_data["indexes"]):
            assert index_name == indexes[i].name
