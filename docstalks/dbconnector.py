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
        while collection_name in collections_names:
            delete = ''
            while delete not in ['Yes', 'y', 'yes', 'No', 'n', 'no']:
                print_color(f"Collection '{collection_name}' is alread exists!\n\
Do you want to delete the existing colletion and recreate \
a new one with the same name?", 'red')
                delete = input("Yes/no: ")
            if delete in ['Yes', 'y', 'yes']:
                client.delete_collection(collection_name=collection_name)
                collections_names.remove(collection_name)
            elif delete in ['No', 'n', 'no']:
                collection_name = str(input("Provide a new collection name: "))
            
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
    collection_name = check_collection_exists_in_qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embedding_lenght=embedding_lenght,
        distance=distance,
    )
    return qdrant_client, collection_name


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
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=doc.metadata['uuid'][i],
                    vector=doc.embeddings[i],
                    payload=doc.metadata,
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
