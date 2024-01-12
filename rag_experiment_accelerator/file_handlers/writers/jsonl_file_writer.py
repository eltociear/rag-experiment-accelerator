import json

# from rag_experiment_accelerator.artifact.models.typing import T

from rag_experiment_accelerator.file_handlers.writers.file_writer import FileWriter


class JsonlFileWriter(FileWriter):
    def _write(self, path: str, data, **kwargs):
        with open(path, "a") as file:
            file.write(json.dumps(data, **kwargs) + "\n")
