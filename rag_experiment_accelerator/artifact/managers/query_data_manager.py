from rag_experiment_accelerator.artifact.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.artifact.managers.artifact_manager import (
    ArtifactManager,
)
from rag_experiment_accelerator.artifact.models.query_data import QueryData
from rag_experiment_accelerator.artifact.loaders.typing import U
from rag_experiment_accelerator.artifact.writers.jsonl_writer import JsonlWriter


class QueryDataManager(ArtifactManager[QueryData, JsonlLoader, JsonlWriter]):
    def __init__(self, output_directory: str) -> None:
        super().__init__(
            class_to_load=QueryData,
            directory=output_directory,
            loader=JsonlLoader(),
            writer=JsonlWriter(),
        )

    def get_output_filename(self, index_name: str) -> str:
        return f"eval_output_{index_name}.jsonl"

    def get_output_filepath(self, index_name: str) -> str:
        return f"{self._directory}/{self.get_output_filename(index_name)}"

    def archive(self, index_name: str):
        output_filename = self.get_output_filename(index_name)
        return super().archive_artifact(output_filename)

    def save(self, data: QueryData, index_name: str):
        output_filename = self.get_output_filename(index_name)
        super().save_artifact(data, output_filename)
