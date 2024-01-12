import json
import pathlib

from rag_experiment_accelerator.file_handlers.writers.file_writer import FileWriter


class JsonFileWriter(FileWriter):
    def _write(self, path: str, data, **kwargs):
        with open(path, "w") as file:
            # json.dump(data, file, indent=4)
            json.dump(data, file, **kwargs)
