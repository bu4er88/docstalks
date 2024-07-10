from qdrant_client import QdrantClient, models
from langchain_core.documents.base import Document
import numpy as np
import base64
from qdrant_client.models import Distance, VectorParams, models
from qdrant_client import QdrantClient
from docstalks.utils import print_color
from copy import deepcopy
    

class QdrandClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client = QdrantClient(url=f"{self.host}:{self.port}")

    def __str__(self):
        return f"QdrantClient(host={self.host}, port={self.port}, \
    embedding_model_name={self.client.embedding_model_name})"

    def get_collections_list(self):
        try:
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            return {"result": "failed get collection list: {e}"}
        
    def create_collection(self, collection_name, embedding_model,
                          distance=models.Distance.COSINE):
        try:
            if collection_name not in self.get_collections_list():
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_model.encode('test').shape[0], 
                        distance=distance))
            return {"result": "collection created: {collection_name}"}
        except Exception as e:
            return {"result": "failed create collection {collection_name}: {e}"}
    
    def delete_collection(self, collection_name):
        try:
            if collection_name not in self.get_collections_list():
                self.client.delete_collection(collection_name=collection_name)
            return {"result": f"collection deleted: {collection_name}"}
        except Exception as e:
            return {"result": f"failed delete collection {collection_name}: {e}"}
        
    def upsert_document(self, 
                        document, 
                        collection_name,
                        methods='default', 
                        use_text_window=False
                        ):
        try:
            parts = len(document.metadata['texts'])
            for i in range(parts):
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

                # upsert data
                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=doc.metadata['uuid'][i],
                            vector=doc.embeddings[i],
                            payload=metadata)
                        ]
                    )
        except Exception as e:
            return {"result": "failed upsert document: {e}"}

    def search(self, query_embedding: list, 
               collection_name: str, limit: int = 5):
        try:
            return self.client.search(
                collection_name, query_embedding, limit)
        except Exception as e:
            return {"result": "search error: {e}"}
    
    def close_connection(self):
        try:
            self.client.close()
            return {"result": "connection was closed"}
        except Exception as e:
            return {"result": "filed closing connection: {e}"}


# def connect_qdrant_host(config, 
#                         embedding_model,
#                         distance=models.Distance.COSINE
#                         ):
#     try:
#         url = f"{config['db_host']}:{config['db_port']}"
#         qdrant_client = QdrantClient(url=url)
        # collection_name = check_collection_exists_in_qdrant(
#             client=qdrant_client,
#             collection_name=config['collection_name'],
#             embedding_lenght=embedding_model.encode('test').shape[0],
#             distance=distance,
#         )
#         return qdrant_client, collection_name
#     except Exception as e:
#         print(f"Failed qdrant connection: {e}")
#         return False, collection_name


def check_collection_exists_in_qdrant(client, 
                                      collection_name,
                                      embedding_lenght,
                                      distance,
                                      ):
    collections_list = client.get_collections().collections
    collections_names = [collection.name for collection in collections_list]
    if collection_name in collections_names:
        print_color(f"Collection '{collection_name}' exists!", 'green')
    else:
        print_color(f"""Collection '{collection_name}' was not found!\
Do you want to create a new empty collection 'default'?.""", 'red')
        result = str(input("Yes/no"))
        if result == 'Yes':
            collection_name = create_qdrant_collection(
                client=client, 
                collection_name=collection_name,
                embedding_lenght=embedding_lenght,
                distance=distance,
                )
    return collection_name

        

def create_qdrant_collection(client, 
                             collection_name,
                             embedding_lenght,
                             distance,
                             ):
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_lenght, 
                distance=distance,
            ),
        )
        return collection_name
    except Exception as e:
        print(f"Failed create qdrant collection: {e}")
        return False

def delete_qdrant_collection(client,
                             collection_name):
    try:
        client.delete_collection(collection_name="{collection_name}")
        return True
    except Exception as e:
        print(f"The collection {collection_name} couldn't be deleted: {e}")
        return False
    

# def add_files_to_qdrant(flist,
#                         config, 
#                         qdrant_client, 
#                         collection_name: str,):
#     if len(collection_name) == 0:
#         collection_name = config['collection_name']
#     for doc in flist:
#         try:    
#             add_document_to_qdrant_db(
#                 document=doc,
#                 client=qdrant_client,
#                 collection_name=collection_name, 
#                 use_text_window=config['use_text_window'],
#                 methods=config['methods'],
#             )
#         except Exception as e:
#             return {"result": "A document was't uploaded: {e}"}



