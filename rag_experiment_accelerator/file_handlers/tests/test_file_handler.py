from abc import ABC
import os

from rag_experiment_accelerator.file_handlers.file_handler import FileHandler


def test_exists_true() -> bool:
    # test this file exists
    path = os.path.realpath(__file__)
    handler = FileHandler()
    exists = handler.exists(path)
    assert exists is True


def test_exists_false() -> bool:
    path = "thisisnotarealpath"
    handler = FileHandler()
    exists = handler.exists(path)
    assert exists is False
