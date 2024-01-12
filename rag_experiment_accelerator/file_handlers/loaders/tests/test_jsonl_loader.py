import json
from rag_experiment_accelerator.file_handlers.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.file_handlers.loaders.tests.fixtures import temp_dir


def test_loads(temp_dir: str):
    test_data = {"test": {"test1": 1, "test2": 2}}
    # write the file
    path = f"{temp_dir}/test.jsonl"
    with open(path, "a") as file:
        file.write(json.dumps(test_data) + "\n")

    # load the file
    loader = JsonlLoader()
    loaded_data = loader.loads(path)

    assert loaded_data == [test_data]


def test_can_handle_true():
    path = "test.jsonl"
    loader = JsonlLoader()
    assert loader.can_handle(path) is True


def test_can_handle_false():
    path = "test.txt"
    loader = JsonlLoader()
    assert loader.can_handle(path) is False
