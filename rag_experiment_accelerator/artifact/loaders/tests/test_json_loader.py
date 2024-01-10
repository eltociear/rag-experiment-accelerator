from rag_experiment_accelerator.artifact.loaders.json_loader import JsonLoader
from rag_experiment_accelerator.artifact.loaders.tests.helpers import TestHelper, helper


def test_load(helper: TestHelper):
    # write the file
    test_data = [{"test": {"test1": 1, "test2": 2}}]
    helper.write_file(test_data, ".json")

    # load the file
    loader = JsonLoader()
    loaded_data = loader.load(helper.path)

    assert loaded_data == test_data


def test_loads(helper: TestHelper):
    # write the file
    test_data = [{"test": {"test1": 1, "test2": 2}}]
    helper.write_file(test_data, ".json")

    # load the file
    loader = JsonLoader()
    loaded_data = loader.loads(helper.path)

    assert loaded_data == test_data


def test_loads_is_always_list(helper: TestHelper):
    # write a file that is not a json list
    test_data = {"test": {"test1": 1, "test2": 2}}
    helper.write_file(test_data, ".json")

    # load the file
    loader = JsonLoader()
    loaded_data = loader.loads(helper.path)

    assert type(loaded_data) == list
    assert loaded_data == [test_data]


def test_can_handle_true():
    loader = JsonLoader()
    assert loader.can_handle(".json") is True


def test_can_handle_false():
    loader = JsonLoader()
    assert loader.can_handle(".txt") is False
