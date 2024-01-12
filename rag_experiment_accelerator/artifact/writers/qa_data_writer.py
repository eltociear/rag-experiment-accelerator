import os
from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.file_handlers.loaders.jsonl_loader import JsonlLoader

from rag_experiment_accelerator.artifact.models.qa_data import QAData
from rag_experiment_accelerator.file_handlers.writers.jsonl_file_writer import (
    JsonlFileWriter,
)


class QADataWriter(ArtifactWriter[QAData]):
    def __init__(self, path: str) -> None:
        self.filename = os.path.basename(path)
        super().__init__(
            directory=os.path.dirname(path),
            writer=JsonlFileWriter(),
        )

    def save(self, data: QAData):
        super().save_artifact(data, self.filename)

    def save_all(self, data: list[QAData]):
        for d in data:
            self.save(d)

    def get_write_path(self) -> str:
        return f"{self.directory}/{self.filename}"

    def handle_archive(self):
        return super().archive_artifact(self.filename)
