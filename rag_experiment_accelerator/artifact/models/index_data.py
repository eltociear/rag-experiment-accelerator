from typing import Self
from rag_experiment_accelerator.artifact.models.artifact import Artifact


class IndexData(Artifact):
    def __init__(
        self,
        prefix: str = None,
        chunk_size: int = None,
        overlap: int = None,
        embedding_model_name: str = None,
        ef_construction: int = None,
        ef_search: int = None,
        name: str = None,
    ):
        if name is not None:
            self.name = name
        else:
            for attr in [
                prefix,
                chunk_size,
                overlap,
                embedding_model_name,
                ef_construction,
                ef_search,
            ]:
                if attr is None:
                    raise ValueError(f"{attr} cannot be None if name is None")
            self.name = f"{prefix}-{chunk_size}-{overlap}-{embedding_model_name.lower()}-{ef_construction}-{ef_search}"

    @classmethod
    def create(cls, data: dict | str) -> Self:
        if type(data) == dict:
            return IndexData(**data)
        else:
            return IndexData(name=data)
