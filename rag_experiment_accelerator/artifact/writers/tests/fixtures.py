import os
import shutil
import pytest


@pytest.fixture()
def temp_dirname():
    dir = "/tmp/" + "rag_writer_tests"
    yield dir
    if os.path.exists(dir):
        shutil.rmtree(dir)
