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


import ollama
# response = ollama.chat(model='llama2-uncensored', messages=[
#   {
#     'role': 'system',
#     'content': 'You are an AI assistant.',
#   },
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])


if __name__=='__main__':

    config = Config(config_name="config.yaml").config
    bot_name = config['bot_name']
    limit = config['limit']
    embedding_model = EmbeddingModel(config)
    retriever = QdrantDBCRetriever(config)
    
    llm = LLM(
        config=config, 
        chatbot_name=bot_name,
        prompts_path="/Users/eugene/Desktop/docstalks/docstalks/chat/prompts.yaml",)

    print(config)
    print(f"Setting up the {bot_name} pipeline...")

    while True:
        question = input("Ask your question: ")
        query_embedding = embedding_model.encode(text=question)
        context, sources = retriever.retrieve(query_embedding, limit=limit, query_filter=None)

        response = ollama.chat(model='llama3', messages=[
          {
            'role': 'system',
            'content': "You are AI assistant answering the Question using only the provided Context.\
        Do not try to make up. Answer always in Russian.\
        If context in another language you do always translate you answer in Russian.\
        Your answer must be as short and clear as it possible. Use bullet points and stucturize your answer to sound professional.",
          },
          {
            'role': 'user',
            'content': f"Question: {question}.\n\nContext:\n{context}",
          },
        ])

        answer = response['message']['content']
        print("Answer: ", end='')
        stream_text(answer)

        print('\n')
        print('Sources:')
        for i, src in enumerate(sources):
            print(f"[{i}] {src}")
        print()



