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


print(f"templates_dir: {templates_dir}")


from src.config import CLIENT_ID, CLIENT_SECRET

from docstalks.config import load_config 
from rag_web import embedding_model, retriever, llm, config

from fastapi.templating import Jinja2Templates


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


@app.get("/rag/")
async def read_item(question: Union[str, None] = None):
    query_embedding = embedding_model.encode(text=question)
    context, sources = retriever.retrieve(query_embedding, limit=config['limit'])
    user_message, system_message = llm.generate_llm_input(
        question=question, 
        context=context,)
    answer = llm.openai_answer(user_message, system_message)
    return {"answer": answer, "sources": sources}


async def process_pdf(file_path: str):
    # Implement your PDF processing logic here
    print(f"Processing PDF file: {file_path}")
    # For example, extract text, convert to images, etc.
    from time import sleep
    sleep(3)
    pass

@app.get("/")
async def read_index():
    return {"message": "Go to /static/index.html to upload PDF files"}


@app.post("/upload-pdf/")
async def upload_pdf(files: List[UploadFile] = File):
    
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
    return {"info": "PDF files processed successfully"}



