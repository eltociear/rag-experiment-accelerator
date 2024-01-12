from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.artifact.models.typing import T
from rag_experiment_accelerator.file_handlers.writers.csv_file_writer import (
    CsvFileWriter,
)
import pandas as pd


class EvalDataWriter(ArtifactWriter[CsvFileWriter]):
    def __init__(
        self,
        output_directory: str,
    ) -> None:
        super().__init__(
            directory=output_directory,
            writer=CsvFileWriter(),
        )

    def save_artifact(
        self, df: pd.DataFrame, filename: str, index: bool = False, **kwargs
    ):
        # set up directory structure
        self._writer.prepare_write(self.directory)

        # write file
        path = f"{self.directory}/{filename}"
        self._writer.write(path=path, data=df, index=index, **kwargs)

    def handle_archive(self):
        # onlt archive if directory exists
        if self._writer.exists(self.directory):
            # get all filenames in directory
            filenames = self._writer.list_filenames(self.directory)
            # archive all files
            for filename in filenames:
                super().archive_artifact(filename)
