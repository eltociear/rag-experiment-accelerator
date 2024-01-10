import json

from rag_experiment_accelerator.artifact.writers.writer import Writer


class JsonWriter(Writer):
    def write(self, path: str, data):
        with open(path, "w") as file:
            json.dump(data, file, indent=4)
