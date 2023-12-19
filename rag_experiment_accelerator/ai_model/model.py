
from abc import ABC, abstractmethod

class AIModel(ABC):
    """
    Abstract base class for LLM models.
    """

    def __init__(self, index_id: str, name: str, **kwargs) -> None:
        super().__init__()
        self.index_id = index_id
        self.name = name

    @abstractmethod
    def try_retrieve_model(self):
        """
        Abstract method that tries to retrieve the LLM model.
        """
        pass
