import json
import os
import uuid

import pytest


class TestHelper:
    def __init__(self, temp_file: str = "/tmp/" + uuid.uuid4().__str__()) -> None:
        self.temp_file = temp_file
        self.path = None

    def write_file(self, data, ext: str):
        self.path = self.temp_file + ext
        with open(self.path, "a") as file:
            file.write(json.dumps(data) + "\n")

    def cleanup(self):
        if self.path:
            os.remove(self.path)


@pytest.fixture()
def helper():
    helper = TestHelper()
    yield helper
    helper.cleanup()
