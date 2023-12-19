from abc import abstractmethod
from typing import Optional

from rag_experiment_accelerator.ai_model.azure_openai_model import AzureOpenAIModel
from rag_experiment_accelerator.ai_model.embedding.model import EmbeddingModel


class AzureOpenAIEmbeddingModel(EmbeddingModel, AzureOpenAIModel):
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

    def __init__(self, dimension: Optional[int], **kwargs) -> None:
        if dimension is None:
            dimension = 1536
        super().__init__(dimension=dimension, tags=["embeddings", "inference"], **kwargs)


    @abstractmethod
    def generate_embedding(self, text: str) -> list:
        """
        method to generate embeddings for a given chunk of text.
        Args:
            chunk (str): The input text chunk for which the embedding needs to be generated.
        Returns:
            list: The generated embedding as a list.
        """

        response = self._client.embeddings.create(
            input=text,
            model=self._deployment_name
        )

        embedding = response.data[0].embedding
        return embedding
