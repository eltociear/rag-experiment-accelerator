import json

from rag_experiment_accelerator.artifact.writers.writer import Writer


class JsonlWriter(Writer):
    def write(self, path: str, data):
        with open(path, "a") as file:
            file.write(json.dumps(data) + "\n")
