import os
import shutil
import time
from typing import Generic
from rag_experiment_accelerator.artifact.loaders.factory import LoaderFactory
from rag_experiment_accelerator.artifact.models.typing import T
from rag_experiment_accelerator.artifact.loaders.typing import U
from rag_experiment_accelerator.artifact.writers.typing import V
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class ArtifactManager(Generic[T, U, V]):
    def __init__(
        self, class_to_load: type[T], directory: str, loader: U, writer: V
    ) -> None:
        self._directory = directory
        self._archive_dir = f"{self._directory}/archive"
        self._class_to_load: type[T] = class_to_load
        self._class_to_load.from_dict
        self._loader_factory = LoaderFactory()
        self._loader = loader
        self._writer = writer

    def _try_make_dir(self, dir: str, exist_ok: bool = True):
        if not os.path.exists(dir):
            try:
                os.makedirs(dir, exist_ok=exist_ok)
            except Exception as e:
                logger.error(
                    f"Unable to create the directory: {dir}. Please ensure"
                    " you have the proper permissions to create the directory."
                )
                raise e

    def init_directory_structure(self) -> None:
        self._try_make_dir(self._directory)

    def archive_artifact(self, filename: str):
        src = f"{self._directory}/{filename}"
        if os.path.exists(src):
            self._try_make_dir(self._archive_dir)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            dest = f"{self._archive_dir}/{timestamp}-{filename}"
            shutil.copyfile(src, dest)
            os.remove(src)
            return dest

    def save_artifact(self, data: T, filename: str) -> None:
        self._try_make_dir(self._directory)
        src = f"{self._directory}/{filename}"
        self._writer.write(src, data.to_dict())

    def load_artifacts(self, filename: str) -> list[T]:
        src = f"{self._directory}/{filename}"
        content = self._loader.loads(path=src)
        data_load: list[T] = [self._class_to_load.from_dict(d) for d in content]
        return data_load
