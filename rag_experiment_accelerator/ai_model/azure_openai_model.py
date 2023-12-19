from openai import AzureOpenAI
from rag_experiment_accelerator.ai_model.model import AIModel
from rag_experiment_accelerator.auth.credentials import OpenAICredentials

from rag_experiment_accelerator.utils.logging import get_logger
logger = get_logger(__name__)

class AzureOpenAIModel(AIModel):
    """
    A class representing an OpenAI model.
    Args:
        deployment_name (str): The name of the model.
        tags (list[str]): A list of tags associated with the model.
        credentials (OpenAICredentials): An instance of the OpenAICredentials class.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    Attributes:
        _tags (list[str]): A list of tags associated with the model.
        _client (AzureOpenAI): An instance of the AzureOpenAI class.
    Methods:
        try_retrieve_model: Tries to retrieve the model and performs necessary checks.
    """

    def __init__(self, deployment_name: str, tags: list[str], credentials: OpenAICredentials, **kwargs) -> None:
        super().__init__(**kwargs)
        self._deployment_name: str = deployment_name
        self._tags: list[str] = tags
        self._client: AzureOpenAI = AzureOpenAI(
                azure_endpoint=credentials.OPENAI_ENDPOINT, 
                api_key=credentials.OPENAI_API_KEY,  
                api_version=credentials.OPENAI_API_VERSION
            )


    def try_retrieve_model(self):
        """
        Tries to retrieve the model and performs necessary checks.
        Returns:
            model: The retrieved model.
        Raises:
            ValueError: If the model is not ready or does not have the required capabilities.
        """
        model = self._client.models.retrieve(model=self._deployment_name)
        if model.status != "succeeded":
            logger.critical(f"Model {self._deployment_name} is not ready.")
            raise ValueError(f"Model {self._deployment_name} is not ready.")
        for tag in self._tags:
            if not model.capabilities[tag]:
                raise ValueError(
                    f"Model {self._deployment_name} does not have the {tag} capability."
                )
        return model

        
    def name(self):
        return self._deployment_name
