from rag_experiment_accelerator.artifact.loaders.json_loader import JsonLoader
from rag_experiment_accelerator.artifact.managers.artifact_manager import (
    ArtifactManager,
)
from rag_experiment_accelerator.artifact.models.index import Index
from rag_experiment_accelerator.artifact.writers.json_writer import JsonWriter


class IndexDataManager(ArtifactManager[Index, JsonLoader, JsonWriter]):
    def __init__(
        self, output_directory: str, filename: str = "generated_index_names.jsonl"
    ) -> None:
        super().__init__(
            class_to_load=Index,
            directory=output_directory,
            writer=JsonWriter(),
            loader=JsonLoader(),
        )
        self._filename = filename
        self.output_filepath = f"{self._directory}/{filename}"

    def save(self, indexes: list[Index]):
        index_dict = {"indexes": [i.name for i in indexes]}
        self._try_make_dir(self._directory)
        self._writer.write(self.output_filepath, index_dict)
