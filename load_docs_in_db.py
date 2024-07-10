import torch 
from docstalks.utils import (convert_text_to_embedding,
                             stream_text,
                             create_document,
                             print_color,
                             split_webpage_into_documents,
                             read_url_in_document,
                             create_document_from_url,
                             )
from docstalks.dbconnector import QdrandClient
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

config = load_config("config.yaml")

embedding_model = SentenceTransformer(
    config['embedding_model']
)
use_text_window = config['use_text_window']
chunk_length = config['chunk_length']

print_color("********** Config: **********", "yellow")
print_color(config, 'yellow')
print_color("*****************************", "yellow")


def add_files_to_qdrant(flist,
                        config, 
                        qdrant_client, 
                        collection_name: str,):
    if len(collection_name) == 0:
        collection_name = config['collection_name']

    for doc in tqdm(flist):
        try:    
            # add_document_to_qdrant_db(
            #     document=doc,
            #     client=qdrant_client,
            #     collection_name=collection_name, 
            #     use_text_window=config['use_text_window'],
            #     methods=config['methods'],
            # )
            pass
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
        default='path',
        # required=True,
        help="Type of the data source",
    )
    parser.add_argument(
        "--source", 
        type=str, 
        # required=True,
        # default="/Users/eugene/Desktop/SoftTeco/danswer/data-softteco/company profiles/",
        default='/Users/eugene/Downloads/Discovery_report.pdf', #"/Users/eugene/Desktop/SoftTeco/danswer/data-softteco/company profiles/",
        help="Path to the documents",
    )
    args = parser.parse_args()
    soruce_type = args.type
    source = args.source

    qdrant_client = QdrandClient(host=config['db_host'], port=config['db_port'])

    # If Source is Path to a folder with PDFs:
    if config['collection_name'] not in qdrant_client.get_collections_list():
        qdrant_client.create_collection(
            collection_name=config['collection_name'],
            embedding_model=embedding_model
        )
    if soruce_type == 'path':
        if source.split('.')[-1] == "pdf":
            fnames = [source]
        else:
            fnames = os.listdir(source)
        print(f"Number of files to upload into the database: {len(fnames)}")
        flist = []
        for fname in fnames:
            try:
                filename = os.path.join(source, fname)
                document = create_document(
                    filename=filename, 
                    config=config, 
                    embedding_model=embedding_model,
                    methods=config['methods'],
                )
                qdrant_client.upsert_document(
                    document=document, 
                    collection_name=config['collection_name'],
                    methods=config['methods']
                )
            except Exception as e:
                print(f"The file was't added to the database: {fname}")
                print(f"Exception: {e}")
        
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
                qdrant_client.upsert_document(
                    document, config, qdrant_client, 
                )

    qdrant_client.close_connection()
    