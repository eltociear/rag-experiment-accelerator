import csv
import pathlib
import pandas as pd
from rag_experiment_accelerator.file_handlers.writers.file_writer import FileWriter


class CsvFileWriter(FileWriter):
    def _write(self, path: str, data: pd.DataFrame, **kwargs):
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, **kwargs)
        else:
            raise Exception(
                f"Unsupported data type {type(data)}. CSVFileWriter can only write pandas.DataFrame"
            )
