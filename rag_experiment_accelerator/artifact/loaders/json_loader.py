import json
from typing import Any
from rag_experiment_accelerator.artifact.loaders.loader import Loader


class JsonLoader(Loader):
    def load(self, path: str) -> Any:
        with open(path, "r") as file:
            content = file.read()
            loaded_data = json.loads(content)
            return loaded_data

    def loads(self, path: str) -> list:
        loaded_data = self.load(path)
        if type(loaded_data) == list:
            return loaded_data
        return [loaded_data]

    def can_handle(self, extension: str):
        return extension == ".json"
