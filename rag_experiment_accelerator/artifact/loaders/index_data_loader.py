from rag_experiment_accelerator.artifact.loaders.artifact_loader import ArtifactLoader
from rag_experiment_accelerator.artifact.models.index_data import IndexData
from rag_experiment_accelerator.file_handlers.loaders.json_loader import JsonLoader


class IndexDataLoader(ArtifactLoader[IndexData]):
    def __init__(
        self, data_dir: str, filename: str = "generated_index_names.json"
    ) -> None:
        super().__init__(
            class_to_load=IndexData,
            directory=data_dir,
            loader=JsonLoader(),
        )
        self.filename = filename
        self.output_filepath = f"{self.directory}/{filename}"

    def load_all(self) -> list[IndexData]:
        return super().load_artifacts(self.filename)
