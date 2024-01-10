import json
import pathlib
import shutil
import tempfile

import pytest
from rag_experiment_accelerator.artifact.managers.qa_data_manager import QADataManager
from rag_experiment_accelerator.artifact.models.qa_data import QAData


class TestHelper:
    def __init__(self, filename: str, test_data: list[QAData] = None) -> None:
        self.test_data = test_data
        self.filename = filename
        self.temp_dir = tempfile.mkdtemp()
        self.filepath = f"{self.temp_dir}/{self.filename}"

    def cleanup(self):
        shutil.rmtree(self.temp_dir)


@pytest.fixture(autouse=True)
def helper():
    test_data = [
        QAData("user_prompt1", "output_prompt1", "context1"),
        QAData("user_prompt2", "output_prompt2", "context2"),
    ]
    filename = "test.jsonl"
    helper = TestHelper(filename=filename, test_data=test_data)
    yield helper
    helper.cleanup()


def test_save(helper: TestHelper):
    qa_manager = QADataManager(helper.filepath)
    # save the data
    for d in helper.test_data:
        qa_manager.save(d)
    # check that the data was saved
    assert pathlib.Path(helper.filepath).exists()
    with open(helper.filepath, "r") as file:
        for i, line in enumerate(file):
            data_dict = json.loads(line)
            assert data_dict["user_prompt"] == helper.test_data[i].user_prompt
            assert data_dict["output_prompt"] == helper.test_data[i].output_prompt
            assert data_dict["context"] == helper.test_data[i].context


def test_loads(helper: TestHelper):
    # save the data to a temp file so it can be loaded
    qa_manager = QADataManager(helper.filepath)
    for d in helper.test_data:
        qa_manager.save(d)

    # load the data
    loaded_data = qa_manager.loads()
    for i, d in enumerate(loaded_data):
        # assertions
        assert d.user_prompt == helper.test_data[i].user_prompt
        assert d.output_prompt == helper.test_data[i].output_prompt
        assert d.context == helper.test_data[i].context


def test_get_filepath(helper: TestHelper):
    qa_manager = QADataManager(helper.filepath)
    assert qa_manager.get_filepath() == f"{helper.temp_dir}/{helper.filename}"


def test_archive(helper: TestHelper):
    qa_manager = QADataManager(helper.filepath)
    # write test data
    for d in helper.test_data:
        qa_manager.save(d)
    # archive the file
    archive_filepath = qa_manager.archive()
    # assertions
    assert pathlib.Path(f"{helper.temp_dir}/archive").exists()
    assert pathlib.Path(archive_filepath).exists()
    assert not pathlib.Path(f"{helper.temp_dir}/{helper.filename}").exists()


def test_archive_no_op(helper: TestHelper):
    qa_manager = QADataManager(helper.filepath)
    archive_filepath = qa_manager.archive()
    assert archive_filepath is None
    assert not pathlib.Path(f"{helper.temp_dir}/archive").exists()
    assert not pathlib.Path(f"{helper.temp_dir}/{helper.filename}").exists()
