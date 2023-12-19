import os
import json
from dotenv import load_dotenv

from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.run.args import get_directory_arg

load_dotenv(override=True)

from rag_experiment_accelerator.init_Index.create_index import create_acs_index
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.embedding.gen_embeddings import generate_embedding
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.nlp.preprocess import Preprocess

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def run(config_dir: str) -> None:
    """
    Runs the main experiment loop, which chunks and uploads data to Azure Cognitive Search indexes based on the configuration specified in the Config class.
    
    Returns:
        None
    """
    config = Config(config_dir)
    pre_process = Preprocess()

    service_endpoint = config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT
    key = config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY

    try:
        os.makedirs(config.artifacts_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Unable to create the '{config.artifacts_dir}' directory. Please ensure you have the proper permissions and try again")
        raise e
    index_dict = {"indexes": []}

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for embedding_model in config.embedding_models:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{embedding_model.index_id}-{ef_construction}-{ef_search}"
                        logger.info(
                            f"{config.NAME_PREFIX}-{config_item}-{overlap}-{embedding_model.index_id}-{ef_construction}-{ef_search}"
                        )
                        create_acs_index(
                            service_endpoint,
                            index_name,
                            key,
                            embedding_model.dimension,
                            ef_construction,
                            ef_search,
                            config.LANGUAGE["analyzers"],
                        )
                        index_dict["indexes"].append(index_name)

    index_output_file = f"{config.artifacts_dir}/generated_index_names.jsonl"
    with open(index_output_file, "w") as index_name:
        json.dump(index_dict, index_name, indent=4)

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for embedding_model in config.embedding_models:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{embedding_model.index_id}-{ef_construction}-{ef_search}"
                        docs = load_documents(
                            config.DATA_FORMATS, config.data_dir, config_item, overlap
                        )

                        upload_data(
                            docs=docs,
                            service_endpoint=service_endpoint,
                            index_name=index_name,
                            search_key=key,
                            chat_model_name=config.CHAT_MODEL_NAME,
                            temperature=config.TEMPERATURE,
                            embedding_model=embedding_model
                        )
