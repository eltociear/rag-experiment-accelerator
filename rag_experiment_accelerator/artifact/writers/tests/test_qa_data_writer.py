import json
import pathlib

from rag_experiment_accelerator.artifact.models.qa_data import QAData
from rag_experiment_accelerator.artifact.writers.qa_data_writer import QADataWriter
from rag_experiment_accelerator.artifact.writers.tests.fixtures import temp_dirname


def test_save(temp_dirname: str):
    filename = "test.jsonl"
    temp_path = f"{temp_dirname}/{filename}"
    writer = QADataWriter(temp_path)

    test_data = [
        QAData("user_prompt1", "output_prompt1", "context1"),
        QAData("user_prompt2", "output_prompt2", "context2"),
    ]
    # save the data
    for d in test_data:
        writer.save(d)

    # check that the data was saved
    assert pathlib.Path(temp_path).exists()
    with open(temp_path, "r") as file:
        for i, line in enumerate(file):
            data_dict = json.loads(line)
            assert data_dict["user_prompt"] == test_data[i].user_prompt
            assert data_dict["output_prompt"] == test_data[i].output_prompt
            assert data_dict["context"] == test_data[i].context


def test_get_write_path(temp_dirname: str):
    filename = "test.jsonl"
    temp_path = f"{temp_dirname}/{filename}"
    writer = QADataWriter(temp_path)
    assert writer.get_write_path() == f"{temp_dirname}/{filename}"


def test_handle_archive(temp_dirname: str):
    # write test data
    filename = "test.jsonl"
    temp_path = f"{temp_dirname}/{filename}"
    writer = QADataWriter(temp_path)
    test_data = [
        QAData("user_prompt1", "output_prompt1", "context1"),
        QAData("user_prompt2", "output_prompt2", "context2"),
    ]
    for d in test_data:
        writer.save(d)

    # archive the file
    archive_filepath = writer.handle_archive()

    # assertions
    assert pathlib.Path(f"{temp_dirname}/archive").exists()
    assert pathlib.Path(archive_filepath).exists()
    assert not pathlib.Path(f"{temp_dirname}/{filename}").exists()


def test_handle_archive_no_op(temp_dirname: str):
    filename = "test.jsonl"
    temp_path = f"{temp_dirname}/{filename}"
    writer = QADataWriter(temp_path)

    archive_filepath = writer.handle_archive()

    assert archive_filepath is None
    assert not pathlib.Path(f"{temp_dirname}/archive").exists()
    assert not pathlib.Path(temp_path).exists()
