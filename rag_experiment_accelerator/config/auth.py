import os
import openai
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def _mask_string(s: str, start: int = 2, end: int = 2, mask_char: str = "*") -> str:
    """
    Masks a string by replacing some of its characters with a mask character.

    Args:
        s (str): The string to be masked.
        start (int): The number of characters to keep at the beginning of the string.
        end (int): The number of characters to keep at the end of the string.
        mask_char (str): The character to use for masking.

    Returns:
        str: The masked string.

    Raises:
        None
    """
    if s == "" or s is None:
        return ""

    if len(s) <= start + end:
        return s[0] + mask_char * (len(s) - 1)

    return (
        s[:start] + mask_char * (len(s) - start - end) + s[-end:]
        if end > 0
        else s[:start] + mask_char * (len(s) - start)
    )


def _get_env_var(var_name: str, critical: bool, mask: bool) -> str:
    """
    Get the value of an environment variable.

    Args:
        var_name (str): The name of the environment variable to retrieve.
        critical (bool): Whether or not the function should raise an error if the variable is not set.
        mask (bool): Whether or not to mask the value of the variable in the logs.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the `critical` parameter is True and the environment variable is not set.
    """
    var = os.getenv(var_name, None)
    if var is None:
        logger.critical(f"{var_name} environment variable not set.")
        if critical:
            raise ValueError(f"{var_name} environment variable not set.")
    else:
        text = var if not mask else _mask_string(var)
        logger.info(f"{var_name} set to {text}")
    return var


class AzureSearchCredentials:
    """
    A class representing the credentials required to access an Azure Search service.

    Attributes:
        AZURE_SEARCH_SERVICE_ENDPOINT (str): The endpoint URL of the Azure Search service.
        AZURE_SEARCH_ADMIN_KEY (str): The admin key required to access the Azure Search service.
    """

    def __init__(
        self,
        azure_search_service_endpoint: str,
        azure_search_admin_key: str,
    ) -> None:
        self.AZURE_SEARCH_SERVICE_ENDPOINT = azure_search_service_endpoint
        self.AZURE_SEARCH_ADMIN_KEY = azure_search_admin_key

    @classmethod
    def from_env(cls) -> "AzureSearchCredentials":
        """
        Creates an instance of AzureSearchCredentials using environment variables.

        Returns:
            AzureSearchCredentials: An instance of AzureSearchCredentials.
        """
        return cls(
            azure_search_service_endpoint=_get_env_var(
                var_name="AZURE_SEARCH_SERVICE_ENDPOINT",
                critical=False,
                mask=False,
            ),
            azure_search_admin_key=_get_env_var(
                var_name="AZURE_SEARCH_ADMIN_KEY",
                critical=False,
                mask=True,
            ),
        )


class AzureMLCredentials:
    """
    A class representing the credentials required to access an Azure Machine Learning workspace.

    Attributes:
        SUBSCRIPTION_ID (str): The subscription ID of the Azure account.
        WORKSPACE_NAME (str): The name of the Azure Machine Learning workspace.
        RESOURCE_GROUP_NAME (str): The name of the resource group containing the Azure Machine Learning workspace.
    """

    def __init__(
        self,
        subscription_id: str,
        workspace_name: str,
        resource_group_name: str,
    ) -> None:
        self.SUBSCRIPTION_ID = subscription_id
        self.WORKSPACE_NAME = workspace_name
        self.RESOURCE_GROUP_NAME = resource_group_name

    @classmethod
    def from_env(cls) -> "AzureMLCredentials":
        """
        Creates an instance of AzureMLCredentials using environment variables.

        Returns:
            AzureMLCredentials: An instance of AzureMLCredentials.
        """
        return cls(
            subscription_id=_get_env_var(
                var_name="AML_SUBSCRIPTION_ID",
                critical=False,
                mask=True,
            ),
            workspace_name=_get_env_var(
                var_name="AML_WORKSPACE_NAME",
                critical=False,
                mask=False,
            ),
            resource_group_name=_get_env_var(
                var_name="AML_RESOURCE_GROUP_NAME",
                critical=False,
                mask=False,
            ),
        )



class OpenAICredentials:
    """
    A class to store OpenAI credentials.

    Attributes:
        OPENAI_API_TYPE (str): The type of OpenAI API to use.
        OPENAI_API_KEY (str): The API key for the OpenAI API.
        OPENAI_API_VERSION (str): The version of the OpenAI API to use.
        OPENAI_ENDPOINT (str): The endpoint for the OpenAI API.

    Methods:
        __init__(self, openai_api_type: str, openai_api_key: str, openai_api_version: str, openai_endpoint: str) -> None:
            Initializes the OpenAICredentials object.
        from_env(cls) -> "OpenAICredentials":
            Creates an OpenAICredentials object from environment variables.
        _set_credentials(self) -> None:
            Sets the OpenAI credentials.
    """

    def __init__(
        self,
        openai_api_type: str,
        openai_api_key: str,
        openai_api_version: str,
        openai_endpoint: str,
    ) -> None:
        """
        Initializes the OpenAICredentials object.

        Args:
            openai_api_type (str): The type of OpenAI API to use.
            openai_api_key (str): The API key for the OpenAI API.
            openai_api_version (str): The version of the OpenAI API to use.
            openai_endpoint (str): The endpoint for the OpenAI API.

        Raises:
            ValueError: If openai_api_type is not 'azure' or 'open_ai'.
        """
        if openai_api_type is not None and openai_api_type not in ["azure", "open_ai"]:
            logger.critical("OPENAI_API_TYPE must be either 'azure' or 'open_ai'.")
            raise ValueError("OPENAI_API_TYPE must be either 'azure' or 'open_ai'.")

        self.OPENAI_API_TYPE = openai_api_type
        self.OPENAI_API_KEY = openai_api_key
        self.OPENAI_API_VERSION = openai_api_version
        self.OPENAI_ENDPOINT = openai_endpoint

        self.set_credentials()

    @classmethod
    def from_env(cls) -> "OpenAICredentials":
        """
        Creates an OpenAICredentials object from environment variables.

        Returns:
            OpenAICredentials: The OpenAICredentials object.
        """
        return cls(
            openai_api_type=_get_env_var(
                var_name="OPENAI_API_TYPE",
                critical=False,
                mask=False,
            ),
            openai_api_key=_get_env_var(
                var_name="OPENAI_API_KEY", critical=False, mask=True
            ),
            openai_api_version=_get_env_var(
                var_name="OPENAI_API_VERSION",
                critical=False,
                mask=False,
            ),
            openai_endpoint=_get_env_var(
                var_name="OPENAI_ENDPOINT",
                critical=False,
                mask=True,
            ),
        )

    def set_credentials(self) -> None:
        """
        Sets the OpenAI credentials.
        """
        openai.api_type = self.OPENAI_API_TYPE
        openai.api_key = self.OPENAI_API_KEY
        logger.info(f"OpenAI API key set to {_mask_string(openai.api_key)}")
        if self.OPENAI_API_TYPE == "open_ai":
                openai.api_version = None
                if self.OPENAI_ENDPOINT is None:
                    openai.api_base = "https://api.openai.com/v1"
                else:
                    openai.api_base = self.OPENAI_ENDPOINT
        elif self.OPENAI_API_TYPE == "azure":
            openai.api_version = self.OPENAI_API_VERSION
            openai.api_base = self.OPENAI_ENDPOINT