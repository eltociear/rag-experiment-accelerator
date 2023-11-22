import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_dir)

import os
import csv
import pandas as pd
import numpy as np
import yaml
import argparse
import json
import argparse
from azure.keyvault.secrets import SecretClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from rag_experiment_accelerator.init_Index.create_index import create_acs_index

def dump_df_to_mltable(
    pdfs_data_source: str,
    target_path: str,
    keyvault_name: str,
    num_partitions: int

):
    print(os.getcwd())

    KVUri = f"https://{keyvault_name}.vault.azure.net"

    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=KVUri, credential=credential)

    os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"] =  client.get_secret("AZURE-SEARCH-SERVICE-ENDPOINT").value
    os.environ["AZURE_SEARCH_ADMIN_KEY"] =  client.get_secret("AZURE-SEARCH-ADMIN-KEY").value

    service_endpoint =  client.get_secret("AZURE-SEARCH-SERVICE-ENDPOINT").value
    key =  client.get_secret("AZURE-SEARCH-ADMIN-KEY").value

    folder_path = pdfs_data_source
    pdfs_file = 'file_names.csv'

    with open('./search_config.json', 'r') as json_file:
        data = json.load(json_file)

    chunk_sizes = data["chunking"]["chunk_size"]
    overlap_size = data["chunking"]["overlap_size"]

    embedding_dimensions = data["embedding_dimension"]
    efConstructions = data["ef_construction"]
    efsearch = data["ef_search"]
    name_prefix = data["name_prefix"]
    analyzers = data["language"]["analyzers"]

    all_index_config = "generated_index_names.txt"

    files = os.listdir(folder_path)

    with open(pdfs_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_name'])  
        for file_name in files:
            if file_name != ".DS_Store":
                writer.writerow([file_name])

    print(f"File names have been written to {pdfs_file}.")

    indexes = []

    for config_item in chunk_sizes:
        for overlap in overlap_size:
            for dimension in embedding_dimensions:
                for efConstruction in efConstructions:
                    for efs in efsearch:
                        index_name = f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efs}"
                        print(f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efs}")
                        create_acs_index(service_endpoint,index_name, key, dimension, efConstruction, efs,analyzers)
                        indexes.append(index_name) 

    with open("index_file.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index_name'])  
        for index in indexes:
            writer.writerow([index])

    print(f"File names have been written to index_file.csv.")

    files_df = pd.read_csv(pdfs_file)
    index_df = pd.read_csv("index_file.csv")

    result_df = pd.DataFrame({
        'index_name': np.repeat(index_df['index_name'].values, len(files_df)),
        'file_name': np.tile(files_df['file_name'].values, len(index_df))
    })

    divisor = int(num_partitions)

    result_df['divider'] =  [(i // divisor) + 1 for i in range(len(result_df))]

    merged_file = 'merged_data.csv'
    result_df.to_csv(merged_file)

    data_by_divider = {}

    with open(merged_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            divider = row['divider']
            if divider not in data_by_divider:
                data_by_divider[divider] = []
            data_by_divider[divider].append(row)
            print(data_by_divider[divider])

    for divider, data in data_by_divider.items():
        folder_name = f'{divider}'
        os.makedirs(os.path.join(os.getcwd(), target_path, folder_name), exist_ok=True)  
        csv_file_path = os.path.join(os.getcwd(), target_path, folder_name, f'data_divider.csv')

        with open(csv_file_path, 'w', newline='') as csv_file:
            fieldnames = data[0].keys()
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(data)

    mltable_path = os.path.join(target_path, "MLTable")
    mltable_dic = {
        "type": "mltable",
        "paths": [],
        "transformations": [
            {
                "read_delimited": {
                    "delimiter": ",",
                    "encoding": "ascii",
                    "header": "all_files_same_headers",
                    "empty_as_string": False,
                    "include_path_column": False,
                },
            },
        ],
    }
    mltable_dic["paths"].append({"pattern": "./**/*.csv"})
    with open(mltable_path, "w") as yaml_file:
        yaml.dump(mltable_dic, yaml_file, default_flow_style=False)


parser = argparse.ArgumentParser()
parser.add_argument("--pdfs_data_source", type=str)
parser.add_argument("--tabular_output_data", type=str)
parser.add_argument("--keyvault_name", type=str)
parser.add_argument("--num_partitions", type=str)

args, _ = parser.parse_known_args()

dump_df_to_mltable( args.pdfs_data_source, args.tabular_output_data, args.keyvault_name, args.num_partitions)