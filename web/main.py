from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import sys
from typing import Union, Optional

dir_1 = Path(__file__).parent.parent
dir_2 = Path(__file__).parent.parent
sys.path.append(str(dir_1))
sys.path.append(str(dir_2))

from docstalks.config import load_config 
from rag_web import embedding_model, retriever, llm, config

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

def print_1(a=1):
    print("********************** text **********************")


@app.get("/rag")
def read_item(question: Union[str, None] = None):
    query_embedding = embedding_model.encode(text=question)
    context, sources = retriever.retrieve(query_embedding, limit=config['limit'])
    user_message, system_message = llm.generate_llm_input(
        question=question, 
        context=context,)
    answer = llm.openai_answer(user_message, system_message)
    return {"answer": answer, "sources": sources}
