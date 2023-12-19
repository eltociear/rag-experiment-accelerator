from rag_experiment_accelerator.ai_model.embedding.azure_openai_model import AzureOpenAIEmbeddingModel
from rag_experiment_accelerator.ai_model.embedding.sentence_transformer_model import SentenceTransformerEmbeddingModel


class EmbeddingModelFactory:
    """
    Factory class for creating embedding models based on the specified embedding type.
    """

    @staticmethod
    def create(embedding_type: str, name: str, **kwargs):
        """
        Create an embedding model based on the specified embedding type.
        Args:
            embedding_type (str): The type of embedding model to create. Must be one of 'openai' or 'huggingface'.
            model_name (str): The name of the model.
            dimension (int): The dimension of the embedding.
            openai_creds (OpenAICredentials): The OpenAI credentials.
        Returns:
            An instance of the embedding model based on the specified embedding type.
        Raises:
            ValueError: If the specified embedding type is invalid.
        """
        if embedding_type == "azure_openai":
            return AzureOpenAIEmbeddingModel(deployment_name=name, name=name, **kwargs)
        elif embedding_type == "sentence_transformers":
            return SentenceTransformerEmbeddingModel(name=name, **kwargs)
        else:
            raise ValueError(f"Invalid embedding type: {embedding_type}. Must be one of 'azure_openai', 'sentence_transformers'")