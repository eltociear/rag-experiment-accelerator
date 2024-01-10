from abc import ABC
from typing import Self


class Artifact(ABC):
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(**data)
