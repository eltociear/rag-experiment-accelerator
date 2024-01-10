import pathlib
from typing import Generic
from rag_experiment_accelerator.artifact.loaders.exceptions import (
    UnsupportedFileFormatException,
)
from rag_experiment_accelerator.artifact.loaders.json_loader import JsonLoader
from rag_experiment_accelerator.artifact.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.artifact.loaders.typing import U


class LoaderFactory(Generic[U]):
    @staticmethod
    def get_loader(filename: str) -> U:
        # get file extension
        ext = pathlib.Path(filename).suffix

        loaders: list[U] = [JsonlLoader(), JsonLoader()]
        for loader in loaders:
            # return loader if can handle the file extension
            if loader.can_handle(ext):
                return loader
        # raise exception if no loader can handle the file extension
        raise UnsupportedFileFormatException(f"Unsupported file format: {ext}")
