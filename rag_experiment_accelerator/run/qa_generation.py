import os

from dotenv import load_dotenv

from rag_experiment_accelerator.artifact.writers.qa_data_writer import QADataWriter
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.data_assets.data_asset import create_data_asset
from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.ingest_data.acs_ingest import generate_qna
from rag_experiment_accelerator.utils.auth import get_default_az_cred
from rag_experiment_accelerator.utils.logging import get_logger

load_dotenv(override=True)


logger = get_logger(__name__)


def run(config_dir: str):
    """
    Runs the main experiment loop for the QA generation process using the provided configuration and data.

    Returns:
        None
    """
    config = Config(config_dir)
    azure_cred = get_default_az_cred()
    all_docs = load_documents(config.DATA_FORMATS, config.data_dir, 2000, 0)

    try:
        os.makedirs(config.artifacts_dir, exist_ok=True)
    except Exception as e:
        logger.error(
            f"Unable to create the '{config.artifacts_dir}' directory. Please"
            " ensure you have the proper permissions and try again"
        )
        raise e

    qa_data = generate_qna(all_docs, config.AZURE_OAI_CHAT_DEPLOYMENT_NAME)
    writer = QADataWriter(config.qa_data_file_path)
    writer.handle_archive()
    writer.save_all(qa_data)

    create_data_asset(
        config.qa_data_file_path,
        "eval_data",
        azure_cred,
        config.AzureMLCredentials,
    )
