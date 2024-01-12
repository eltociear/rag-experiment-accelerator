from rag_experiment_accelerator.artifact.loaders.artifact_loader import ArtifactLoader
from rag_experiment_accelerator.file_handlers.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.artifact.models.qa_data import QAData


class QADataLoader(ArtifactLoader[QAData]):
    def __init__(self, filepath: str) -> None:
        loader = JsonlLoader()
        directory = loader.get_dirname(filepath)
        self._filename = self.loader.get_basename(filepath)
        super().__init__(
            class_to_load=QAData,
            directory=directory,
            loader=loader,
        )

    def load_all(self) -> list[QAData]:
        data: list[QAData] = super().load_artifacts(self.filename)
        return data
