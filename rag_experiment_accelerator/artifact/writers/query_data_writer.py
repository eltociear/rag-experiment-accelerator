from rag_experiment_accelerator.artifact.common.query_data_handler import (
    QueryDataHandler,
)
from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.artifact.models.query_data import QueryData
from rag_experiment_accelerator.file_handlers.writers.jsonl_file_writer import (
    JsonlFileWriter,
)


class QueryDataWriter(ArtifactWriter[QueryData], QueryDataHandler):
    def __init__(self, output_dir: str) -> None:
        super().__init__(
            directory=output_dir,
            writer=JsonlFileWriter(),
        )

    def handle_archive(self, index_name: str):
        output_filename = self.get_output_filename(index_name)
        return super().archive_artifact(output_filename)

    def save(self, data: QueryData, index_name: str):
        output_filename = self.get_output_filename(index_name)
        super().save_artifact(data, output_filename)
