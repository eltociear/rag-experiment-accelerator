from dotenv import load_dotenv

from rag_experiment_accelerator.artifact.loaders.index_data_loader import (
    IndexDataLoader,
)
from rag_experiment_accelerator.artifact.models.index_data import IndexData
from rag_experiment_accelerator.artifact.writers.index_data_writer import (
    IndexDataWriter,
)
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.init_Index.create_index import create_acs_index
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from rag_experiment_accelerator.utils.logging import get_logger

load_dotenv(override=True)


logger = get_logger(__name__)


def run(config_dir: str, data_dir: str = "data") -> None:
    """
    Runs the main experiment loop, which chunks and uploads data to Azure Cognitive Search indexes based on the configuration specified in the Config class.

    Returns:
        None
    """
    config = Config(config_dir, data_dir)
    pre_process = Preprocess()

    service_endpoint = config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT
    key = config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY

    indexes: list[IndexData] = []
    for chunk_size in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for embedding_model in config.embedding_models:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index = IndexData(
                            config.NAME_PREFIX,
                            chunk_size,
                            overlap,
                            embedding_model.name,
                            ef_construction,
                            ef_search,
                        )

                        logger.info(f"Creating Index with name: {index.name}")
                        create_acs_index(
                            service_endpoint,
                            index.name,
                            key,
                            embedding_model.dimension,
                            ef_construction,
                            ef_search,
                            config.LANGUAGE["analyzers"],
                        )
                        indexes.append(index)

    writer = IndexDataWriter(config.index_data_dir)
    loader = IndexDataLoader(config.index_data_dir)

    prev_indexes = loader.load_all()
    if prev_indexes is not None:
        writer.handle_archive(indexes, prev_indexes)

    writer.save_all(indexes)

    for chunk_size in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            all_docs = load_documents(
                config.DATA_FORMATS, config.data_dir, chunk_size, overlap
            )
            for embedding_model in config.embedding_models:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index = IndexData(
                            config.NAME_PREFIX,
                            chunk_size,
                            overlap,
                            embedding_model.name,
                            ef_construction,
                            ef_search,
                        )
                        data_load = []
                        for docs in all_docs:
                            chunk_dict = {
                                "content": docs.page_content,
                                "content_vector": embedding_model.generate_embedding(
                                    chunk=str(pre_process.preprocess(docs.page_content))
                                ),
                            }
                            data_load.append(chunk_dict)
                        upload_data(
                            chunks=data_load,
                            service_endpoint=service_endpoint,
                            index_name=index.name,
                            search_key=key,
                            embedding_model=embedding_model,
                            azure_oai_deployment_name=config.AZURE_OAI_CHAT_DEPLOYMENT_NAME,
                        )
