import json
from rag_experiment_accelerator.file_handlers.loaders.json_loader import JsonLoader
from rag_experiment_accelerator.file_handlers.loaders.tests.fixtures import temp_dir


def test_loads(temp_dir: str):
    # write the file
    test_data = [{"test": {"test1": 1, "test2": 2}}]
    path = f"{temp_dir}/test.json"
    with open(path, "a") as file:
        file.write(json.dumps(test_data) + "\n")

    # load the file
    loader = JsonLoader()
    loaded_data = loader.loads(path)

    assert loaded_data == test_data


def test_loads_is_always_list(temp_dir: str):
    # write a file that is not a json list
    test_data = {"test": {"test1": 1, "test2": 2}}
    path = f"{temp_dir}/test.json"
    with open(path, "a") as file:
        file.write(json.dumps(test_data) + "\n")

    # load the file
    loader = JsonLoader()
    loaded_data = loader.loads(path)

    assert type(loaded_data) == list
    assert loaded_data == [test_data]


def test_can_handle_true():
    path = "test.json"
    loader = JsonLoader()
    assert loader.can_handle(path) is True


def test_can_handle_false():
    path = "test.txt"
    loader = JsonLoader()
    assert loader.can_handle(path) is False
