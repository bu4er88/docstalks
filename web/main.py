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


@app.get("/rag")
async def read_item(question: Union[str, None] = None):
    print(1)
    query_embedding = embedding_model.encode(text=question)
    context, sources = retriever.retrieve(query_embedding, limit=config['limit'])
    user_message, system_message = llm.generate_llm_input(
        question=question, 
        context=context,)
    print(context)
    answer = llm.openai_answer(user_message, system_message)
    print(answer)
    return {"answer": answer} #{"answer": answer, "sources": sources}
