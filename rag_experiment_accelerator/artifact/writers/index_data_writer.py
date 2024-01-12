import os
from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.artifact.models.index_data import IndexData
from rag_experiment_accelerator.file_handlers.writers.json_file_writer import (
    JsonFileWriter,
)


class IndexDataWriter(ArtifactWriter[IndexData]):
    def __init__(
        self, output_directory: str, filename: str = "generated_index_names.json"
    ) -> None:
        super().__init__(
            directory=output_directory,
            writer=JsonFileWriter(),
        )
        self.filename = filename
        self.output_filepath = f"{self.directory}/{filename}"

    def save_all(self, indexes: list[IndexData]):
        # get index names
        index_names = [i.name for i in indexes]

        # set up directory structure
        self._prepare_write()

        # write file
        self._writer.write(self.output_filepath, index_names, indent=4)

    def _should_archive(
        self, new_indexes: list[IndexData], prev_indexes: list[IndexData]
    ):
        # make sure the file exists
        if self._writer.exists(self.output_filepath):
            # if the number of indexes is different, then we should archive
            if len(new_indexes) != len(prev_indexes):
                return True

            # if the names are different, then we should archive
            if [i.name for i in prev_indexes] != [i.name for i in new_indexes]:
                return True

        return False

    def handle_archive(
        self, new_indexes: list[IndexData], prev_indexes: list[IndexData]
    ):
        if self._should_archive(new_indexes, prev_indexes):
            return super().archive_artifact(self.filename)
