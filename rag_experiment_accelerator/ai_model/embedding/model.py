from abc import abstractmethod
from rag_experiment_accelerator.ai_model.model import AIModel


class EmbeddingModel(AIModel):
    """
    Base class for embedding models.
    Args:
        model_name (str): The name of the embedding model.
        dimension (int): The dimension of the embeddings.
    Attributes:
        dimension (int): The dimension of the embeddings.
    Methods:
        generate_embedding(chunk: str) -> list: Abstract method to generate embeddings for a given chunk of text.
    """

    def __init__(self, dimension: int,**kwargs) -> None:
            super().__init__(**kwargs)
            self.dimension = dimension

    @abstractmethod
    def generate_embedding(self, chunk: str) -> list:
            """
            abstract method to generate embeddings for a given chunk of text.
            Args:
                chunk (str): The input text chunk for which the embedding needs to be generated.
            Returns:
                list: The generated embedding as a list.
            """
            pass