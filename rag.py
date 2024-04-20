# from tqdm import tqdm
# from typing import Optional, List, Tuple
# import torch 
from docstalks.utils import (convert_text_to_embedding,
                             stream_text,
                             create_document,)
from docstalks.dbconnector import (initialize_qdrant_client,
                                   add_document_to_qdrant_db,
                                   search_in_qdrant_db,)
from chat.llm import (read_yaml_to_dict,
                      generate_prompt_template,
                      openai_answer,)
from qdrant_client import models
from qdrant_client import QdrantClient
from docstalks.config import load_config 
from sentence_transformers import SentenceTransformer, util
import logging
import os
from tqdm import tqdm
from openai import OpenAI
import yaml



class Config:

    def __init__(self, config_name: str) -> None:
        self.config = load_config(config_name)
        # self.embedding_model_name = self.config['embedding_model_name'] #"config.yaml")
        # self.collection_name = self.config['collection_name'] #"config.yaml")
        # self.db_host = self.config['db_host']
        # self.db_port = self.config['db_port']

    def __str__(self) -> str:
        print("""Config:
              confing_name: {config_name}
              self.embedding_model_name: {self.embedding_model_name}
              self.collection_name: {self.collection_name}
              """)
        return
        
    def __repr__(self) -> str:
        print(self.__str__())
        return


class EmbeddingModel:

    def __init__(self, config) -> None:
        self.embedding_model_name = config['embedding_model_name']
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
      
    def encode(self, text):
        return self.embedding_model.encode(
            sentences=text, 
            convert_to_tensor=False,
            convert_to_numpy=True
            ).tolist()
    

class QdrantDBCRetriever:
    
    def __init__(self, config) -> None:
        self.collection_name = config['collection_name']
        self.qdrant_client = QdrantClient(host=config['db_host'], port=config['db_port'])
    
    def retrieve(self, embedding, limit=10):
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit)
        context = [res.payload["text"].strip() for res in results]
        context = '. '.join(context)
        sources = set([res.payload["filename"] for res in results])
        return context, list(sources)

    def filter(self, filter: list):
        #TODO: add logic for filtering in the qdrant database
        pass


class LLM:

    def __init__(self, config, chatbot_name='DocsTalks', prompts_path='chat/prompts.yaml',
                 model_name="gpt-3.5-turbo") -> None:
        self.api_key = config['openai_api_key']
        self.llm = OpenAI(api_key=self.api_key)
        self.chatbot_name = chatbot_name
        self.model_name = model_name
        self.prompts_path = prompts_path
        self.prompts = self.collect_prompts()
        self.openai_answer('', '')

    def collect_prompts(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_directory, self.prompts_path)
        with open(full_path, 'r') as file:
            prompt_dict = yaml.safe_load(file)
        prompts = [
            prompt for prompt in prompt_dict['prompts'] \
            if self.chatbot_name in prompt['name']
        ]
        return prompts[0]

    def generate_llm_input(self, question, context) -> list:
        system_message = self.prompts['system']
        message_template = self.prompts['task']
        message_template += "\nQuestion: {question}\n\nContext:\n{context}\n\nHelpful Answer:"    
        user_message = message_template.format(question=question, context=context)
        return [user_message, system_message]

    def openai_answer(self, user_message, system_message):

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                ],
                temperature=0.1,
                # max_tokens=64,
                # top_p=1
            )
        return response.choices[0].message.content


if __name__=='__main__':

    config = Config(config_name="config.yaml").config
    embedding_model = EmbeddingModel(config)
    retriever = QdrantDBCRetriever(config)
    llm = LLM(config, chatbot_name='DocsTalks')

    while True:
        question = input("Ask your question: ")
        query_embedding = embedding_model.encode(text=question)
        context, sources = retriever.retrieve(query_embedding)
        user_message, system_message = llm.generate_llm_input(question=question, context=context)
        answer = llm.openai_answer(user_message, system_message)
        print("Answer: ", end='')
        stream_text(answer)

        print('\n')
        print('Sources:')
        for i, src in enumerate(sources):
            print(f"[{i}] {src}")
        print()



    # user_message, system_message = generate_prompt_template(
    #     question, 
    #     context, 
    #     chat_category_name, 
    #     yaml_dict,
    #     )


    # TODO: solve a problem! It alway answers "I am here to assist you. What would you like to know?"

    # answer = openai_answer(
    #     client=client,
    #     question=question,
    #     context=context,
    #     user_message=user_message,
    #     system_message=system_message,
    # )

    # stream_text(answer)