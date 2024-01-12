from rag_experiment_accelerator.artifact.common.query_data_handler import (
    QueryDataHandler,
)
from rag_experiment_accelerator.artifact.loaders.artifact_loader import ArtifactLoader
from rag_experiment_accelerator.artifact.models.query_data import QueryData
from rag_experiment_accelerator.file_handlers.loaders.jsonl_loader import JsonlLoader


class QueryDataLoader(ArtifactLoader[QueryData], QueryDataHandler):
    def __init__(self, output_dir: str) -> None:
        super().__init__(
            class_to_load=QueryData,
            directory=output_dir,
            loader=JsonlLoader(),
        )

    def load_all(self, index_name: str) -> list[QueryData]:
        path = self.get_output_filename(index_name)
        return super().load_artifacts(path)
