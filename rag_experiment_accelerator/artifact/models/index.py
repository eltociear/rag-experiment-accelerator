from typing import Self
from rag_experiment_accelerator.artifact.models.artifact import Artifact


class Index(Artifact):
    def __init__(
        self,
        prefix: str,
        chunk_size: int,
        overlap: int,
        embedding_model_name: str,
        ef_construction: int,
        ef_search: int,
    ):
        self.name = f"{prefix}-{chunk_size}-{overlap}-{embedding_model_name.lower()}-{ef_construction}-{ef_search}"
