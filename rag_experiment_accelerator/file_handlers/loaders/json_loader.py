import json
import pathlib
from typing import Any
from rag_experiment_accelerator.file_handlers.loaders.loader import Loader


class JsonLoader(Loader):
    def loads(self, path: str, **kwargs) -> list:
        with open(path, "r") as file:
            content = file.read()
            loaded_data = json.loads(content, **kwargs)

        if type(loaded_data) == list:
            return loaded_data
        return [loaded_data]

    def can_handle(self, path: str):
        ext = pathlib.Path(path).suffix
        return ext == ".json"
