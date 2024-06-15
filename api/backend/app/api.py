from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import sys
from typing import Union, Optional
import os 
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi.staticfiles import StaticFiles

backend_dir = Path(__file__).parent.parent
docstalks_dir = Path(__file__).parent.parent.parent.parent
templates_dir = os.path.join(docstalks_dir, 'web/frontend/templates/')

sys.path.append(str(backend_dir))
sys.path.append(str(docstalks_dir))
sys.path.append(templates_dir)

print(templates_dir)

from src.config import CLIENT_ID, CLIENT_SECRET

from docstalks.config import load_config 
from rag_web import embedding_model, retriever, llm, config

from fastapi.templating import Jinja2Templates


app = FastAPI()
origins = [
    "http://localhost:8000",
    "localhost:8000",
]
app.add_middleware(SessionMiddleware, secret_key="add any string MAAAAN!...")
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



@app.get('/')
def index(request: Request):
    return templates.TemplateResponse(
        name='index.html',
        context={'request': request}
    )


# handle coomunication with Front-end framework
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Get Route
@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message":   "Using the app you need do /rag?question='add your question here'"}



@app.get("/rag")
def read_item(question: Union[str, None] = None):
    query_embedding = embedding_model.encode(text=question)
    context, sources = retriever.retrieve(query_embedding, limit=config['limit'])
    user_message, system_message = llm.generate_llm_input(
        question=question, 
        context=context,)
    answer = llm.openai_answer(user_message, system_message)
    return {"answer": answer, "sources": sources}
