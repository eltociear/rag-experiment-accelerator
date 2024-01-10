import os
from rag_experiment_accelerator.artifact.loaders.jsonl_loader import JsonlLoader

# from rag_experiment_accelerator.artifact.loaders.typing import U
from rag_experiment_accelerator.artifact.managers.artifact_manager import (
    ArtifactManager,
)
from rag_experiment_accelerator.artifact.models.qa_data import QAData
from rag_experiment_accelerator.artifact.writers.jsonl_writer import JsonlWriter


class QADataManager(ArtifactManager[QAData, JsonlLoader, JsonlWriter]):
    def __init__(self, filepath: str) -> None:
        self._directory = os.path.dirname(filepath)
        self._filename = os.path.basename(filepath)
        super().__init__(
            class_to_load=QAData,
            directory=self._directory,
            writer=JsonlWriter(),
            loader=JsonlLoader(),
        )

    def save(self, data: QAData):
        super().save_artifact(data, self._filename)

    def loads(self) -> list[QAData]:
        data: list[QAData] = super().load_artifacts(self._filename)
        return data

    def get_filepath(self) -> str:
        return f"{self._directory}/{self._filename}"

    def archive(self):
        return super().archive_artifact(self._filename)
