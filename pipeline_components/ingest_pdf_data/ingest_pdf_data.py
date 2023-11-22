import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_dir)

import os
import json

from rag_experiment_accelerator.doc_loader.documentLoader import load_documents
from rag_experiment_accelerator.embedding.gen_embeddings import generate_embedding
from rag_experiment_accelerator.ingest_data.acs_ingest import upload_data
from rag_experiment_accelerator.nlp.preprocess import Preprocess
import argparse
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from spacy import cli

#cli.download("en_core_web_lg")


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyvault_name", type=str)
    parser.add_argument("--pdfs_data_source", type=str)
    parser.add_argument("--search_config", type=str)
    
    args, _ = parser.parse_known_args()
    global pdfs_data_source
    pdfs_data_source = args.pdfs_data_source
    search_config = args.search_config


    global data

    existing_env_var_value = os.environ.get('CONDA_DEFAULT_ENV')
    if existing_env_var_value:
        new_env_var_value = "/azureml-envs/" + existing_env_var_value + "/share/tessdata/"
        os.environ['TESSDATA_PREFIX'] = new_env_var_value
        print("New environment variable value:", os.environ['TESSDATA_PREFIX'])
    else:
        print("EXISTING_ENV_VAR is not defined.")

    with open(search_config, 'r') as json_file:
        data = json.load(json_file)

    keyVaultName = args.keyvault_name
    KVUri = f"https://{keyVaultName}.vault.azure.net"

    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)

    global AZURE_SEARCH_SERVICE_ENDPOINT
    global AZURE_SEARCH_ADMIN_KEY
    global OPENAI_ENDPOINT
    global OPENAI_API_KEY
    global OPENAI_API_VERSION
    global SUBSCRIPTION_ID
    global WORKSPACE_NAME
    global RESOURCE_GROUP_NAME
    os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"] =  client.get_secret("AZURE-SEARCH-SERVICE-ENDPOINT").value
    os.environ["AZURE_SEARCH_ADMIN_KEY"] =  client.get_secret("AZURE-SEARCH-ADMIN-KEY").value
    os.environ["OPENAI_ENDPOINT"] =  client.get_secret("OPENAI-ENDPOINT").value
    os.environ["OPENAI_API_KEY"] =  client.get_secret("OPENAI-API-KEY").value
    os.environ["OPENAI_API_VERSION"] =  client.get_secret("OPENAI-API-VERSION").value
    os.environ["SUBSCRIPTION_ID"] =  client.get_secret("SUBSCRIPTION-ID").value
    os.environ["WORKSPACE_NAME"] =  client.get_secret("WORKSPACE-NAME").value
    os.environ["RESOURCE_GROUP_NAME"] =  client.get_secret("RESOURCE-GROUP-NAME").value

    AZURE_SEARCH_SERVICE_ENDPOINT  =  client.get_secret("AZURE-SEARCH-SERVICE-ENDPOINT").value
    AZURE_SEARCH_ADMIN_KEY =  client.get_secret("AZURE-SEARCH-ADMIN-KEY").value
    OPENAI_ENDPOINT =  client.get_secret("OPENAI-ENDPOINT").value
    OPENAI_API_KEY =  client.get_secret("OPENAI-API-KEY").value
    OPENAI_API_VERSION =  client.get_secret("OPENAI-API-VERSION").value
    SUBSCRIPTION_ID =  client.get_secret("SUBSCRIPTION-ID").value
    WORKSPACE_NAME =  client.get_secret("WORKSPACE-NAME").value
    RESOURCE_GROUP_NAME =  client.get_secret("RESOURCE-GROUP-NAME").value


def run(input_data):
    print(input_data)

    pre_process = Preprocess()


    service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")  
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

    chat_model_name=data["chat_model_name"]
    temperature = data["openai_temperature"]
    embedding_model_name = data["embedding_model_name"]
    data_formats = data["data_formats"]

    index_dict = {}
    index_dict["indexes"] = []
                        
    for index in range(len(input_data)):
        index_name = input_data.loc[index, 'index_name']
        items = index_name.split('-')
        print(items)
        prefix, chunk_size, chunk_overlap, dimension, efConstruction, efsearch = items
        pdf_name = input_data.loc[index, 'file_name']

        if os.path.exists(os.path.join(pdfs_data_source, pdf_name)):
            print(os.path.join(pdfs_data_source, pdf_name))
            all_docs = load_documents(data_formats, os.path.join(pdfs_data_source, pdf_name), int(chunk_size), int(chunk_overlap))
            data_load = []
            for docs in all_docs:
                chunk_dict = {
                    "content": docs.page_content,
                    "content_vector": generate_embedding(
                        size=dimension,
                        chunk=str(pre_process.preprocess(docs.page_content)),
                        model_name=embedding_model_name
                    )
                }
                data_load.append(chunk_dict)
            upload_data(
                chunks=data_load,
                service_endpoint=service_endpoint,
                index_name=index_name,
                search_key=key,
                dimension=dimension,
                chat_model_name=chat_model_name,
                embedding_model_name=embedding_model_name,
                temperature=temperature
            )
    return []

