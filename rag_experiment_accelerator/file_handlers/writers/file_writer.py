from abc import ABC, abstractmethod
import os
import pathlib
import shutil
import os
from os.path import isfile, join
from rag_experiment_accelerator.file_handlers.file_handler import FileHandler

# from rag_experiment_accelerator.artifact.models.typing import T

# from rag_experiment_accelerator.file_handlers.writers.writer import Writer

from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class FileWriter(FileHandler):
    def _try_make_dir(self, dir: str, exist_ok: bool = True):
        try:
            os.makedirs(dir, exist_ok=exist_ok)
        except Exception as e:
            logger.error(
                f"Unable to create the directory: {dir}. Please ensure"
                " you have the proper permissions to create the directory."
            )
            raise e

    def prepare_write(self, dir: str):
        # dest_dir = pathlib.Path(path).parent
        self._try_make_dir(dir)

    @abstractmethod
    def _write():
        pass

    def write(self, path: str, data, **kwargs):
        dir = pathlib.Path(path).parent
        self.prepare_write(dir)
        self._write(path, data, **kwargs)

    def copy(self, src: str, dest: str, **kwargs):
        shutil.copyfile(src, dest, **kwargs)

    def delete(self, src: str):
        if os.path.exists(src):
            os.remove(src)

    def list_filenames(self, dir: str):
        return [f for f in os.listdir(dir) if isfile(join(dir, f))]
