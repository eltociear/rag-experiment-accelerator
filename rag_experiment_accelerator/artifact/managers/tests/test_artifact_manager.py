import json
import os
import pathlib
import shutil
import uuid
import pytest
from rag_experiment_accelerator.artifact.loaders.jsonl_loader import JsonlLoader

from rag_experiment_accelerator.artifact.managers.artifact_manager import (
    ArtifactManager,
)
from rag_experiment_accelerator.artifact.models.artifact import Artifact
from rag_experiment_accelerator.artifact.writers.jsonl_writer import JsonlWriter


class TestHelper:
    def __init__(
        self,
        temp_dir: str = "/tmp/" + uuid.uuid4().__str__(),
        test_data: list[Artifact] = None,
    ) -> None:
        self.test_data = test_data
        self.temp_dir = temp_dir

    def make_temp_dir(self):
        os.makedirs(self.temp_dir)

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


@pytest.fixture()
def helper():
    test_data = [
        Artifact(),
        Artifact(),
    ]
    helper = TestHelper(test_data=test_data)
    yield helper
    helper.cleanup()


def test_init_directory_structure(helper: TestHelper):
    artifact_manager = ArtifactManager(
        Artifact, helper.temp_dir, writer=JsonlWriter(), loader=JsonlLoader()
    )
    artifact_manager.init_directory_structure()
    assert pathlib.Path(helper.temp_dir).exists()


def test_archive_artifact(helper: TestHelper):
    artifact_manager = ArtifactManager(
        Artifact, helper.temp_dir, writer=JsonlWriter(), loader=JsonlLoader()
    )
    # create the file to archive
    helper.make_temp_dir()
    path = f"{helper.temp_dir}/test.jsonl"
    with open(path, "w") as file:
        file.write(json.dumps(helper.test_data[0].to_dict()) + "\n")

    # archive the artifact
    archive_filepath = artifact_manager.archive_artifact("test.jsonl")

    # archive dir exists
    assert pathlib.Path(f"{helper.temp_dir}/archive").exists()
    # archive file exists
    assert pathlib.Path(archive_filepath).exists()
    # original file does not exist
    assert not pathlib.Path(f"{helper.temp_dir}/test.jsonl").exists()


def test_archive_artifact_no_op(helper: TestHelper):
    artifact_manager = ArtifactManager(
        Artifact, helper.temp_dir, writer=JsonlWriter(), loader=JsonlLoader()
    )

    # archive non-existing artifact
    archive_filepath = artifact_manager.archive_artifact("test.jsonl")

    # did not archive anything
    assert archive_filepath is None
    # archive dir does not exist, since it is a new temp dir
    assert not pathlib.Path(f"{helper.temp_dir}/archive").exists()


def test_load_artifacts_jsonl(helper: TestHelper):
    artifact_manager = ArtifactManager(
        Artifact, helper.temp_dir, writer=JsonlWriter(), loader=JsonlLoader()
    )
    # write the file to load
    helper.make_temp_dir()
    path = f"{helper.temp_dir}/test.jsonl"
    with open(path, "w") as file:
        data = [{}, {}]
        for data_point in data:
            file.write(json.dumps(data_point) + "\n")

    # load the artifacts
    artifacts = artifact_manager.load_artifacts("test.jsonl")

    for a in artifacts:
        assert type(a) == Artifact


def test_save_artifact(helper: TestHelper):
    # setup
    filename = "test.jsonl"
    path = f"{helper.temp_dir}/{filename}"
    artifact_manager = ArtifactManager(
        Artifact, helper.temp_dir, writer=JsonlWriter(), loader=JsonlLoader()
    )

    # save the artifacts
    for a in helper.test_data:
        artifact_manager.save_artifact(a, filename)

    # file exists
    assert pathlib.Path(path).exists()
    # file has the correct number of data points
    assert len(artifact_manager.load_artifacts(filename)) == len(helper.test_data)


def test__try_make_dir(helper: TestHelper):
    artifact_manager = ArtifactManager(
        Artifact, helper.temp_dir, writer=JsonlWriter(), loader=JsonlLoader()
    )
    artifact_manager._try_make_dir(helper.temp_dir)
    assert pathlib.Path(helper.temp_dir).exists()
