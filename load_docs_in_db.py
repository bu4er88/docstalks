# from tqdm import tqdm
# from typing import Optional, List, Tuple
# import torch 
from docstalks.utils import (convert_text_to_embedding,
                             stream_text,
                             create_document,
                             print_color,
                             split_webpage_into_documents,
                             read_url_in_document,
                             create_document_from_url,
                             )
from docstalks.dbconnector import (initialize_qdrant_client,
                                   add_document_to_qdrant_db,
                                   search_in_qdrant_db,)
from chat.llm import (read_yaml_to_dict,
                      generate_prompt_template,
                      openai_answer,)
from qdrant_client import models
from qdrant_client import QdrantClient
from docstalks.config import load_config 
from sentence_transformers import SentenceTransformer, util
import logging
import os
from tqdm import tqdm
from openai import OpenAI
from docstalks.config import load_config 
from argparse import ArgumentParser

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")

config = load_config("config.yaml")

embedding_model = SentenceTransformer(
    config['embedding_model_name']
)

use_text_window = config['use_text_window']
chunk_length = config['chunk_length']

print_color("********** Config: **********", "yellow")
print_color(config, 'yellow')
print_color("*****************************", "yellow")


def init_qdrant(config):
    qdrant_client, collection_name = initialize_qdrant_client(
        embedding_model=embedding_model,
        collection_name=config['collection_name'],
        distance=models.Distance.COSINE,
        url="http://localhost:6333",
    )
    return qdrant_client, collection_name


def add_files_to_qdrant(flist,
                        config, 
                        qdrant_client, 
                        collection_name: str,):
    if len(collection_name) == 0:
        collection_name = config['collection_name']

    for doc in tqdm(flist):
        try:    
            add_document_to_qdrant_db(
                document=doc,
                client=qdrant_client,
                collection_name=collection_name, 
                use_text_window=config['use_text_window'],
                methods=config['methods'],
            )
        except Exception as e:
            print(f"A document was't uploaded..")
            print(f"Exception: {e}")



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--type", 
        choices=['path', 'url'],
        type=str, 
        # default='path',
        default='url',
        # required=True,
        help="Type of the data source",
    )
    parser.add_argument(
        "--source", 
        type=str, 
        # required=True,
        # default="/Users/eugene/Desktop/SoftTeco/danswer/data-softteco/company profiles/",
        default='https://lenalondonphoto.com/', #"/Users/eugene/Desktop/SoftTeco/danswer/data-softteco/company profiles/",
        help="Path to the documents",
    )
    args = parser.parse_args()
    soruce_type = args.type
    source = args.source


    qdrant_client, collection_name = init_qdrant(config)


    # If Source is Path to a folder with PDFs:
    if soruce_type == 'path':
        fnames = os.listdir(source)[:3]
        print(f"Number of files to upload into the database: {len(fnames)}")
        flist = []
        for fname in fnames:
            try:
                filename = os.path.join(source, fname)
                document = create_document(
                    filename=filename, 
                    config=config,
                    chunk_length=150, 
                    embedding_model=embedding_model,
                    methods=config['methods'],
                )
                flist.append(document)
            except Exception as e:
                print(f"The file was't added to the database: {filename}")
                print(f"Exception: {e}")
        add_files_to_qdrant(
            flist, config, qdrant_client, 
            collection_name
        )
        
    # If Source is URL:
    elif soruce_type == 'url':
        documents_dict = split_webpage_into_documents(url=source, recursive=True, ssl_verify=False)
        print(f"Number of files in the uploaded data: {len(documents_dict.keys())}")
        flist = []
        for key in tqdm(documents_dict.keys()):
            if len(documents_dict[key]) > 0:
                document = create_document_from_url(
                    filename=(documents_dict[key]), 
                    config=config,
                    chunk_length=150, 
                    embedding_model=embedding_model,
                    methods=config['methods'],
                )
                flist.append(document)

        add_files_to_qdrant(
            flist, config, qdrant_client, 
            collection_name
        )
