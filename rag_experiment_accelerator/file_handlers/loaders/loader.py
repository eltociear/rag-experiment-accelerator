from abc import ABC, abstractmethod
import os

from rag_experiment_accelerator.file_handlers.file_handler import FileHandler


class Loader(FileHandler):
    @abstractmethod
    def loads(self, path: str, **kwargs) -> list:
        pass

    @abstractmethod
    def can_handle(self, path: str) -> bool:
        pass

    def get_basename(self, path: str) -> str:
        return os.path.basename(path)

    def get_dirname(self, path: str) -> str:
        return os.path.dirname(path)
