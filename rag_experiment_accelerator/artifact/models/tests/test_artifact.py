from typing import Self

from rag_experiment_accelerator.artifact.models.artifact import Artifact


def test_to_dict():
    artifact = Artifact()
    artifact_dict = artifact.to_dict()
    assert artifact_dict == {}


def test_from_dict():
    artifact_dict = {}
    artifact_dict = Artifact.from_dict(artifact_dict)
    assert type(artifact_dict) == Artifact
