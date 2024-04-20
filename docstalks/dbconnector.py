from qdrant_client import QdrantClient, models
from langchain_core.documents.base import Document
import numpy as np
import base64

from qdrant_client.models import Distance, VectorParams, models
from qdrant_client import QdrantClient


def create_qdrant_collection(client, 
                             collection_name,
                             embedding_lenght,
                             distance,
                             ):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_lenght, 
            distance=distance,),)
    return


def check_collection_exists_in_qdrant(client, 
                                      collection_name,
                                      embedding_lenght,
                                      distance,
                                      ):
    collections_list = client.get_collections().collections
    collections_names = [collection.name for collection in collections_list]
    if collection_name in collections_names:
        delete = ''
        while delete not in ['Yes', 'no']:
            delete = input(
                f"""Collection '{collection_name}' is alread 
exists. Do you want to delete the existing 
colletion and recreate a new one with the same name? (Yes/no): """)
        if delete=='Yes':
            client.delete_collection(collection_name=collection_name)
        elif delete=='no':
            collection_name = input("Provide new collection name: ")
            
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

# def binarize_text(text):
#     # Encode text to bytes
#     text_bytes = text.encode('utf-8')
#     # Encode bytes to base64 binary representation
#     binarized_text = base64.b64encode(text_bytes)
#     return binarized_text

# def debinarize_text(binarized_text):
#     # Decode base64 binary representation to bytes
#     text_bytes = base64.b64decode(binarized_text)
#     # Decode bytes to text
#     text = text_bytes.decode('utf-8')
#     return text


# def pad_vector(vector, pad_value=0):
#     """Pads all vectors in a list to the same length.
    
#     Args:
#       vectors: A list of NumPy arrays representing the vectors.
#       target_length: The desired length for all padded vectors.
#       pad_value: The value to use for padding (default: 0).
    
#     Returns:
#       A list of NumPy arrays with all vectors padded to the target length.
#     """
#     target_length = tokenizer.model_max_length
#     padding_length = target_length - len(vector)
#     padding = np.full((padding_length,), pad_value)
#     padded_vector = np.concatenate((vector, padding))
#     return padded_vector


def add_document_to_qdrant_db(document,
                              client,
                              collection_name,
                              use_text_window: bool = False,
                              ):
    # check if the Document type belongs to the llangchain
    for i in range(len(document.metadata['texts'])):
        metadata = {
            'filename': document.metadata['filename'],
            'filetype': document.metadata['filetype'],
            'last_modified': document.metadata['last_modified'],
            'number_of_pages': document.metadata['number_of_pages'],
            'languages': document.metadata['languages']
        }
        if use_text_window:
            metadata['text'] = document.metadata['windows'][i]
        else:
            metadata['text'] = document.metadata['texts'][i]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=document.metadata['uuid'][i],
                    vector=document.embeddings[i],
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
