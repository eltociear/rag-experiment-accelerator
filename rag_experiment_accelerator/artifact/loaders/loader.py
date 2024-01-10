from abc import ABC, abstractmethod


class Loader(ABC):
    @abstractmethod
    def loads(self, path: str) -> list:
        pass

    def can_handle(self, extension: str) -> bool:
        return extension == ".jsonl"
