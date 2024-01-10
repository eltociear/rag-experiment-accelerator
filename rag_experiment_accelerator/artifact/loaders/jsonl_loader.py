import json
from rag_experiment_accelerator.artifact.loaders.loader import Loader


class JsonlLoader(Loader):
    def loads(self, path: str) -> list:
        data_load = []
        with open(path, "r") as file:
            for line in file:
                data = json.loads(line)
                data_load.append(data)
        return data_load

    def can_handle(self, extension: str):
        return extension == ".jsonl"
