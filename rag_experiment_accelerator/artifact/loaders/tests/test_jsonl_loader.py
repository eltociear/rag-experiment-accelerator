from rag_experiment_accelerator.artifact.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.artifact.loaders.tests.helpers import TestHelper, helper


def test_loads(helper: TestHelper):
    test_data = {"test": {"test1": 1, "test2": 2}}
    # write the file
    helper.write_file(test_data, ".jsonl")

    # load the file
    loader = JsonlLoader()
    loaded_data = loader.loads(helper.path)

    assert loaded_data == [test_data]


def test_can_handle_true():
    loader = JsonlLoader()
    assert loader.can_handle(".jsonl") is True


def test_can_handle_false():
    loader = JsonlLoader()
    assert loader.can_handle(".txt") is False
