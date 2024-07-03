from fastapi import (FastAPI,
                     File,
                     UploadFile,)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
from typing import (Union, 
                    List,
                    Optional, 
                    Annotated,)
import os 
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil

backend_dir = Path(__file__).parent.parent
docstalks_dir = Path(__file__).parent.parent.parent.parent
templates_dir = os.path.join(docstalks_dir, 'web/frontend/templates/')
sys.path.append(str(backend_dir))
sys.path.append(str(docstalks_dir))
sys.path.append(templates_dir)

from src.config import CLIENT_ID, CLIENT_SECRET
from docstalks.config import load_config 
from docstalks.dbconnector import (add_files_to_qdrant,
                                   connect_to_qdrant,)
from docstalks.utils import (split_webpage_into_documents,
                             create_document_from_url,)
from rag_web import embedding_model, retriever, llm, config
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from tqdm import tqdm


config = load_config("config.yaml")
embedding_model = SentenceTransformer(config['embedding_model'])
use_text_window = config['use_text_window']
chunk_length = config['chunk_length']
server_host = config['server_host']
server_port = config['server_port']
db_host = config['db_host']
db_port = config['db_port']

app = FastAPI()

origins = [
    "http://localhost:8000",
    "localhost:8000",
    "*"
]

app.add_middleware(SessionMiddleware, secret_key="secret_key")
oauth = OAuth()
oauth.register(
    name='goole',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',\
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    client_kwargs={
        'scope': 'emain openid profile',
        'redirect_uri': 'http://localhost:8000/auth'
    }
)
templates = Jinja2Templates(directory=templates_dir)

# handle coomunication with Front-end framework
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount the static files directory
app.mount(
    "/static", 
    StaticFiles(directory=os.path.join(backend_dir, "static")), 
    name="static"
)


@app.get("/rag")
async def read_item(question: Union[str, None] = None, 
                    filter: Union[dict, None] = None,
                    ):
    query_embedding = embedding_model.encode(question)
    ### TODO: add filtering feature for filtering results
    context, sources = retriever.retrieve(
        query_embedding, limit=config['limit'], query_filter=filter,
    )
    user_message, system_message = llm.generate_llm_input(
        question=question, 
        context=context,
    )
    answer = llm.openai_answer(user_message, system_message)
    return {"answer": answer, "sources": sources}


async def process_pdf(file_path: str):
    # Implement your PDF processing logic here
    print(f"Processing PDF file: {file_path}")
    # For example, extract text, convert to images, etc.
    from time import sleep
    sleep(3)


@app.get("/")
async def read_index():
    return {"message": "ask question with /rag/?question=<your question>"}


@app.post("/upload-pdf")
async def upload_pdf(files: List[UploadFile]):  # = File):
    
    save_dir = os.path.join(backend_dir, "media")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory '{save_dir}' created.")
    else:
        print(f"Directory '{save_dir}' already exists.")
    
    for file in files:
        file_location = os.path.join(save_dir, file.filename)
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(file.file, file_object)
        print(1)
        await process_pdf(file_location)
        # os.remove(file_location)  # Clean up the file after processing
        print(2)
    return {"info": "PDF files uploaded successfully"}


@app.post("/upload-url/")
async def upload_link(url: str,
                      recursive: bool = False,
                      ssl_verify: bool = False,
                      ):
    try:
        qdrant_client, collection_name = connect_to_qdrant(config, embedding_model)
        if not qdrant_client:
            return {"connect_to_qdrant": "error", "collection_name": collection_name}
    except Exception as e:
        return {"Init database error ": str(e)}
    try:
        documents_dict = split_webpage_into_documents(url, recursive, ssl_verify)
        print(f"Number of files in the uploaded data: {len(documents_dict.keys())}")
        
        for key in tqdm(documents_dict.keys()):
            if len(documents_dict[key]) > 0:
                document = create_document_from_url(
                    filename=(documents_dict[key]), 
                    config=config,
                    chunk_length=150, 
                    embedding_model=embedding_model,
                    methods=config['methods'],
                )
                if not isinstance(document, list):
                    document = [document]
                add_files_to_qdrant(
                    document, config, qdrant_client, 
                    collection_name,
                )
        return {"result": "success"}
    except Exception as e:
        return {"result": "Failed url data upload: {e}"}
    