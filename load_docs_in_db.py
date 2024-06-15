# from tqdm import tqdm
# from typing import Optional, List, Tuple
# import torch 
from docstalks.utils import (convert_text_to_embedding,
                             stream_text,
                             create_document,
                             print_color,)
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


def add_to_qdrant(flist,
                  embedding_model,
                  config, 
                  qdrant_client, 
                  collection_name: str = "",
                  ):
    for filename in tqdm(flist):
        try:
            document = create_document(
                filename=filename, 
                config=config,
                chunk_length=150, 
                embedding_model=embedding_model,
                methods=config['methods'],
            )
            if len(collection_name) == 0:
                collection_name = config['collection_name']
            add_document_to_qdrant_db(
                document=document, 
                client=qdrant_client,
                collection_name=collection_name, 
                use_text_window=config['use_text_window'],
                methods=config['methods'],
            )
        except Exception as e:
            print(f"The file was't added to the database: {filename}")
            print(f"Exception: {e}")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source", 
        type=str, 

        # required=True,
        default="/Users/eugene/Desktop/SoftTeco/danswer/data-softteco/company profiles/",

        help="Path to documents",
    )
    args = parser.parse_args()
    source = args.source 

    # If Source is path to a folder with PDFs:
    fnames = os.listdir(source)
    flist = [os.path.join(source, fname) for fname in fnames]
    print(f"Number of files in the uploaded data: {len(flist)}")

    qdrant_client, collection_name = init_qdrant(config)

    add_to_qdrant(flist, embedding_model, config, 
                  qdrant_client, collection_name)
    