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
   



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")

config = load_config("config.yaml")
embedding_model_name = config['embedding_model_name']
collection_name = config['collection_name']
use_text_window = config['use_text_window']
chunk_length = config['chunk_length']

print_color("********** Config: **********", "green")
print_color(config, 'green')
print_color("*****************************", "green")

embedding_model = SentenceTransformer(embedding_model_name)

data_path = "/Users/eugene/Desktop/SoftTeco/danswer/data-softteco"

fnames = os.listdir(data_path)
flist = [os.path.join(data_path, fname) for fname in fnames]
# print(f"INFO: Number of files in the uploaded data: {len(flist)}")


qdrant_client,collection_name = initialize_qdrant_client(
    embedding_model=embedding_model,
    collection_name=collection_name,
    distance=models.Distance.COSINE,
    url="http://localhost:6333",)


for filename in tqdm(flist):
    try:
        document = create_document(
            filename=filename, 
            chunk_length=150, 
            embedding_model=embedding_model,
            method='summaries',)
        add_document_to_qdrant_db(
            document=document, 
            client=qdrant_client, 
            collection_name=collection_name, 
            use_text_window=use_text_window,
            method='summaries',)
    except Exception as e:
        print(f"The file was't added to the database: {filename}")
        print(f"Exception: {e}")
        