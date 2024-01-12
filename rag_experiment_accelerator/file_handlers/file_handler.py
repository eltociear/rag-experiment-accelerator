from abc import ABC
import os


class FileHandler(ABC):
    def exists(self, path: str) -> bool:
        if os.path.exists(path):
            return True
        return False
