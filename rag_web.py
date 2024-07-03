from docstalks.utils import (convert_text_to_embedding,
                             stream_text,
                             create_document,
                             Config,
                             LLM,
                             EmbeddingModel,
                             QdrantDBCRetriever,)
from docstalks.dbconnector import (connect_to_qdrant,
                                   add_document_to_qdrant_db,
                                   search_in_qdrant,)
from chat.llm import (read_yaml_to_dict,
                      generate_prompt_template,
                      openai_answer,)
from qdrant_client import models
from docstalks.config import load_config 
import logging
import os
from tqdm import tqdm
import yaml

from pathlib import Path
import sys

base_dir = str(Path(__file__).parent)
sys.path.append(base_dir)
config_path = os.path.join(base_dir, "config.yaml")

config = Config(config_name=config_path).config
bot_name = config['bot_name']
limit = config['limit']
embedding_model = EmbeddingModel(config)
retriever = QdrantDBCRetriever(config)

llm = LLM(
    config=config, 
    chatbot_name=bot_name,
    prompts_path=os.path.join(base_dir, "chat/prompts.yaml")
)


if __name__=='__main__':

    config = Config(config_name="config.yaml").config
    bot_name = config['bot_name']
    limit = config['limit']
    embedding_model = EmbeddingModel(config)
    retriever = QdrantDBCRetriever(config)
    
    llm = LLM(
        config=config, 
        chatbot_name=bot_name,
        prompts_path="/Users/eugene/Desktop/docstalks/docstalks/chat/prompts.yaml"
    )

    print(config)
    print(f"Setting up the {bot_name} pipeline...")

    # while True:
    #     question = input("Ask your question: ")
    #     query_embedding = embedding_model.encode(text=question)
    #     context, sources = retriever.retrieve(query_embedding, limit=limit)
    #     user_message, system_message = llm.generate_llm_input(
    #         question=question, 
    #         context=context,
    #         )
    #     answer = llm.openai_answer(user_message, system_message)
    #     print("Answer: ", end='')
    #     stream_text(answer)

    #     print('\n')
    #     print('Sources:')
    #     for i, src in enumerate(sources):
    #         print(f"[{i}] {src}")
    #     print()
