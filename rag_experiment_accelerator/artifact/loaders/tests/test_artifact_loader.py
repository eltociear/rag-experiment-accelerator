import json
import os
import shutil
import uuid

import pytest
from rag_experiment_accelerator.artifact.loaders.artifact_loader import ArtifactLoader
from rag_experiment_accelerator.artifact.models.artifact import Artifact
from rag_experiment_accelerator.artifact.writers.artifact_writer import ArtifactWriter
from rag_experiment_accelerator.file_handlers.loaders.jsonl_loader import JsonlLoader
from rag_experiment_accelerator.file_handlers.writers.jsonl_file_writer import (
    JsonlFileWriter,
)
from rag_experiment_accelerator.artifact.loaders.tests.fixtures import temp_dir


def test_loads(temp_dir: str):
    # write artifacts to a file
    filename = "test.jsonl"
    path = f"{temp_dir}/test.jsonl"

    writer = ArtifactWriter(temp_dir, JsonlFileWriter)

    with open(path, "w") as file:
        artifacts = [Artifact(), Artifact()]
        for a in artifacts:
            file.write(json.dumps(a.to_dict()) + "\n")

    # load the file
    loader = ArtifactLoader(
        class_to_load=Artifact, directory=temp_dir, loader=JsonlLoader()
    )
    loaded_data = loader.load_artifacts(filename)

    assert [a.to_dict() for a in loaded_data] == [a.to_dict() for a in artifacts]
    assert [isinstance(a, Artifact) for a in loaded_data]
