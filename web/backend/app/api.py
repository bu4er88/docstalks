from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel
from pathlib import Path
import sys
from typing import Union, Optional

base_dir = Path(__file__).parent.parent.parent.parent
# dir_2 = Path(__file__).parent.parent.parent
sys.path.append(str(base_dir))
# sys.path.append(str(dir_2))

from docstalks.config import load_config 
from rag_web import embedding_model, retriever, llm, config

app = FastAPI()
origins = [
    "http://localhost:3000",
    "localhost:3000",
]

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
    return {"message": "Welcome to docstalks.com!"}



# @app.get("/")
# def read_root():
#     return {"Hello": "World"}



@app.get("/rag")
def read_item(question: Union[str, None] = None):
    query_embedding = embedding_model.encode(text=question)
    context, sources = retriever.retrieve(query_embedding, limit=config['limit'])
    user_message, system_message = llm.generate_llm_input(
        question=question, 
        context=context,)
    answer = llm.openai_answer(user_message, system_message)
    return {"answer": answer, "sources": sources}
