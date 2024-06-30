from qdrant_client import QdrantClient, models
from langchain_core.documents.base import Document
import numpy as np
import base64

from qdrant_client.models import Distance, VectorParams, models
from qdrant_client import QdrantClient
from docstalks.utils import print_color

from copy import deepcopy


def create_qdrant_collection(client, 
                             collection_name,
                             embedding_lenght,
                             distance,
                             ):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_lenght, 
            distance=distance,
        ),
    )


def check_collection_exists_in_qdrant(client, 
                                      collection_name,
                                      embedding_lenght,
                                      distance,
                                      ):
    collections_list = client.get_collections().collections
    collections_names = [collection.name for collection in collections_list]
    if collection_name in collections_names:
        print_color(f"Collection '{collection_name}' already exists!", 'green')
    else:
        print_color(f"Collection '{collection_name}' was not found! Collection name will be 'default'.", 'red')
        collection_name = "default"
        # collection_name = str(input("Provide a new collection name: "))
    create_qdrant_collection(
        client=client, 
        collection_name=collection_name,
        embedding_lenght=embedding_lenght,
        distance=distance,
        )
    return collection_name


def initialize_qdrant_client(embedding_model: str = None,
                             collection_name: str ='default',
                             distance=models.Distance.COSINE,
                             url: str="http://localhost:6333",
                             ):
    qdrant_client = QdrantClient(url=url)
    embedding_lenght = embedding_model.encode('test').shape[0]
    ### TODO: not sure whethere it's useful or not
    collection_name = check_collection_exists_in_qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embedding_lenght=embedding_lenght,
        distance=distance,
    )
    return qdrant_client, collection_name


def connect_to_qdrant_client(embedding_model: str,
                             collection_name: str,
                             distance=models.Distance.COSINE,
                             url: str ="http://localhost:6333",
                             ):
    qdrant_client = QdrantClient(url=url)
    embedding_lenght = embedding_model.encode('test').shape[0]
    collection_name = check_collection_exists_in_qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embedding_lenght=embedding_lenght,
        distance=distance,
    )
    return qdrant_client, collection_name

def add_files_to_qdrant(flist,
                        config, 
                        qdrant_client, 
                        collection_name: str,):
    if len(collection_name) == 0:
        collection_name = config['collection_name']

    for doc in flist:
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


def add_document_to_qdrant_db(document,
                              client,
                              collection_name,
                              use_text_window: bool = False, 
                              methods='default',
                              ):
    # check if the Document type belongs to the llangchain
    for i in range(len(document.metadata['texts'])):
        doc = deepcopy(document)
        if methods != 'default':
            for method in methods:
                doc.metadata[method] = document.metadata[method][i]
        if use_text_window:
            doc.metadata['texts'] = document.metadata['windows'][i]
        else:
            doc.metadata['texts'] = document.metadata['texts'][i]

        # keep only needed required keys
        metadata = deepcopy(doc.metadata)
        metadata['text'] = metadata['texts']
        del metadata['texts']
        metadata['text_window'] = metadata['windows'][i]
        del metadata['windows']
        if 'summary' in methods:
            metadata['summary'] = metadata['summaries']
            del metadata['summaries']
        metadata['uuid'] = metadata['uuid'][i]

        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc.metadata['uuid'][i],
                    vector=doc.embeddings[i],
                    payload=metadata,
                ),
            ],
        )


def search_in_qdrant_db(query_vector: list, 
                        client,
                        collection_name: str, 
                        limit: int = 5,
                        ):
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
        )
    return results
