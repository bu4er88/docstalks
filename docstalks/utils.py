from unstructured.partition.pdf import partition_pdf
import uuid
from time import time, sleep
import random
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, UnstructuredHTMLLoader
)
from sentence_transformers import SentenceTransformer
import yaml
from openai import OpenAI
from qdrant_client import QdrantClient
from typing import Tuple
import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
from .config import load_config 
import os


class Config:

    def __init__(self, config_name: str) -> None:
        self.config = load_config(config_name)
        # self.embedding_model = self.config['embedding_model'] #"config.yaml")
        # self.collection_name = self.config['collection_name'] #"config.yaml")
        # self.db_host = self.config['db_host']
        # self.db_port = self.config['db_port']

    def __str__(self) -> str:
        print("""Config:
              confing_name: {config_name}
              self.embedding_model: {self.embedding_model}
              self.collection_name: {self.collection_name}
              """)
        return
        
    def __repr__(self) -> str:
        print(self.__str__())
        return


class EmbeddingModel:

    def __init__(self, config) -> None:
        self.embedding_model = config['embedding_model']
        self.embedding_model = SentenceTransformer(self.embedding_model)
      
    def encode(self, text):
        return self.embedding_model.encode(
            sentences=text, 
            convert_to_tensor=False,
            convert_to_numpy=True
            ).tolist()
    

class QdrantDBCRetriever:
    """
    method: text / summaries
    """
    def __init__(self, config, ) -> None:
        self.collection_name = config['collection_name']
        self.qdrant_client = QdrantClient(host=config['db_host'], port=config['db_port'])
    
    def retrieve(self, embedding, limit, query_filter):
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit,
            query_filter=query_filter,
        )
        context = [f"Source: {res.payload["filename"]}\n{res.payload["text"].strip()}" for res in results]
        context = '\n\n'.join(context)
        sources = set([res.payload["filename"] for res in results])
        return context, sources

    def filter(self, filter: list):
        #TODO: add logic for filtering in the qdrant database
        pass


class LLM:

    def __init__(self, 
                 config, 
                 chatbot_name='DocsTalks', 
                 prompts_path='prompts.yaml',
                 model_name="gpt-3.5-turbo"
                 ) -> None:
        self.api_key = config['openai_api_key']
        self.llm = OpenAI(api_key=self.api_key)
        self.chatbot_name = chatbot_name
        self.model_name = model_name
        self.prompts_path = prompts_path
        self.prompts = self.collect_prompts()
        self.openai_answer('', '')

    def collect_prompts(self):
        # current_directory = 
        # full_path = os.path.join(current_directory, self.prompts_path)
        with open(self.prompts_path, 'r') as file:
            prompt_dict = yaml.safe_load(file)
        prompts = [
            prompt for prompt in prompt_dict['prompts'] \
            if self.chatbot_name in prompt['name']
        ]
        return prompts[0]

    def generate_llm_input(self, question, context) -> list:
        system_message = self.prompts['system']
        message_template = self.prompts['task']
        message_template += "\nQuestion: {question}\n\nContext:\n\n{context}\n\nHelpful Answer:"    
        user_message = message_template.format(question=question, context=context)
        return [user_message, system_message]

    def openai_answer(self, user_message, system_message):
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                ],
            temperature=0,
            )
        return response.choices[0].message.content


def read_pdf_in_document(file: str, 
                         method: str ='default',
                         chunk_size=1000,
                         chunk_overlap=200,
                         ):
    """Read text from document and split it.

    Args:
        file: string contains path to the file.
        method: 
            'standard' - for reading and naive spliting text;
            'recursive' - for reading and recursive splitting by carracrs.
    Returns:
        Document or List[Document,]
    """
    try:
        if method=='default':
            splits = partition_pdf(
                filename=file,
                include_page_breaks=False,
                strategy='fast',
                infer_table_structure=False,
                include_metadata = True,
                chunking_strategy = None,
            )
            document = splits[0]
            merged_text = " ".join([part.text.strip() for part in splits])
            document.text = merged_text
            document.metadata = {
                'file_directory': document.metadata.file_directory,
                'filename': document.metadata.filename,
                'languages': document.metadata.languages,
                'last_modified': document.metadata.last_modified,
                'number_of_pages': splits[-1].metadata.page_number,
                'filetype': document.metadata.filetype,
                'texts': [],
                'windows': [],
                'classes': [],
                'uuid': [],
            }
            document.category = 'Docstalks'
            return document
        elif method=='recursive':
            loader = UnstructuredPDFLoader(file)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                )
            document = text_splitter.split_documents(data)
            return document
    except Exception as e:
        print(f"🛑 Exception called by 'extract_pdf_elemets_with_unstructured' \
function processing the document: {file}") 
        print(f"Exception: {e}")


def read_url_in_document(documents: list, 
                         method: str ='default',
                         chunk_size=1000,
                         chunk_overlap=200,
                         ):
    """Read text from document and split it.

    Args:
        file: string contains path to the file.
        method: 
            'standard' - for reading and naive spliting texts;
            'recursive' - for reading and recursive splitting by carracrs.
    Returns:
        Document or List[Document,]
    """
    try:
        if method=='default':
            document = documents[0]
            merged_text = " ".join([part.text.strip() for part in documents])
            document.text = merged_text
            document.metadata = {
                'file_directory': 'web',
                'filename': document.metadata.url,
                'languages': document.metadata.languages,
                'last_modified': document.metadata.last_modified,
                'number_of_pages': document.metadata.category_depth,
                'filetype': document.metadata.filetype,
                'texts': [],
                'windows': [],
                'classes': [],
                'uuid': [],
            }
            document.category = 'Docstalks'
            return document
        elif method=='recursive':
            loader = UnstructuredHTMLLoader(documents)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                )
            document = text_splitter.split_documents(data)
            return document
    except Exception as e:
        print(f"🛑 Exception called by 'extract_pdf_elemets_with_unstructured' \
function processing the url scrapping!") 
        print(f"Exception: {e}")


def get_file_extension(file_name: str) -> str:
    file_ext = '.' + file_name.split('.')[-1]
    return file_ext
    

def get_separators(file_name: str) -> str:
    # langchain/libs/langchain/langchain/text_splitter.py
    file_types = {
        ".cpp": "cpp", ".go": "go", ".java": "java", ".kt": "kotlin", 
        ".js": "js", ".ts": "ts", ".php": "php", ".proto": "proto", 
        ".py": "python", ".rst": "rst", ".rb": "ruby", ".rs": "rust", 
        ".scala": "scala", ".swift": "swift", ".md": "markdown", ".txt": "markdown", 
        ".pdf": "markdown", ".tex": "latex", ".html": "html", ".sol": "sol", 
        ".cs": "csharp", ".cobol": "cobol"
    }
    file_ext = get_file_extension(file_name)
    if file_ext not in file_types.keys():
        print(f"INFO: File type is not supported: {file_name}\nSeparators for Markdown will be used as default.")
        file_type = 'markdown'
    else:
        file_type = file_types[file_ext] 
    separators = RecursiveCharacterTextSplitter.get_separators_for_language(file_type)
    return separators


def add_texts_and_windows_to_document(document, chunk_length, overlap):
    overlap = max(chunk_length//10, overlap)
    text = document.text
    words = text.split()
    start = 0
    window_size = int(chunk_length * 0.8)
    window_start = 0
    while start < len(words):
        end = min(start + chunk_length, len(words))
        window_end = min(end + window_size, len(words))
        chunk = " ".join(words[int(start):int(end)])
        window = " ".join(words[int(window_start):int(window_end)])
        document.metadata['texts'].append(chunk)
        document.metadata['windows'].append(window)
        start += chunk_length - overlap
        window_start = start - window_size
    return document


def get_embedding_from_text(text, embedding_model):
    embedding = embedding_model.encode(text, convert_to_tensor=False)
    return embedding.tolist()


def summarize_text(text, llm):
    question = "Summarize the text in the following context."
    user_message, system_message = llm.generate_llm_input(question=question, context=text)
    summary = llm.openai_answer(user_message, system_message)
    return summary


def classify_text(text, llm):
    question = "Summarize the text in the following context."
    user_message, system_message = llm.generate_llm_input(question=question, context=text)
    summary = llm.openai_answer(user_message, system_message)
    return summary


def extract_keywords_from_text(text, llm):
    question = "Extract 1-10 keywords from the following context."
    user_message, system_message = llm.generate_llm_input(question=question, context=text)
    keywords = llm.openai_answer(user_message, system_message)
    return list(keywords.split(' '))


def process_document(document, embedding_model, methods, models):
    if methods == 'default':
        embeddings = [get_embedding_from_text(t, embedding_model) for t in document.metadata['texts']]
    else:
        if (models is None) or \
            (len(models.keys()) == 0) or \
                not isinstance(models, dict):
            raise(f"'process_document()': 'methods' type is not valid: {methods}")
        texts = document.metadata['texts']
        model_result = {}
        for model in models:
            if model not in model_result:
                model_result[model] = [
                    models[model].openai_answer(
                        user_message=models[model].generate_llm_input('', text)[0], 
                        system_message=models[model].generate_llm_input('', text)[1],
                        ) for text in texts
                ]
            document.metadata[model] = model_result[model]
    embeddings = [get_embedding_from_text(text, embedding_model) for text in document.metadata['texts']]
    document.embeddings = embeddings
    return document


def generate_uuid_from_text(text):
    # Define a namespace UUID
    namespace_uuid = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')    
    # Generate a UUID based on the SHA-1 hash of the namespace and the document name
    generated_uuid = str(uuid.uuid5(namespace_uuid, text))
    return generated_uuid


def add_uuid_to_document(document):
    for text in document.metadata['texts']:
        generated_id = generate_uuid_from_text(text)
        document.metadata['uuid'].append(generated_id)
    return document


def convert_text_to_embedding(text, embedding_model):
    embedding = embedding_model.encode(
        sentences=text, 
        convert_to_tensor=False,
        convert_to_numpy=True,
        )
    return embedding.tolist()


def load_documents_with_llangchain(file_path, 
                                   chunk_size=1000,
                                   chunk_overlap=200,
                                   ):
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        )
    splits = text_splitter.split_documents(data)
    return splits


def create_document(filename: str, 
                    config: dict,
                    embedding_model,
                    methods='default',
                    ):
    """
    method: 'texts / summaries'.
    """
    document = read_pdf_in_document(
        filename, chunk_size=config['chunk_length'], 
        chunk_overlap=config['overlap']
    )
    document = add_texts_and_windows_to_document(
        document=document, 
        chunk_length=config['chunk_length'], 
        overlap=config['overlap'],
    )
    if methods=='default':
        llm = None
    else:
        if isinstance(methods, str): 
            methods = list(methods)
        elif isinstance(methods, list):
            models = {}
            for method in methods:
                if method not in models.keys():
                    models[method]= LLM(
                        config=config,
                        chatbot_name=method, 
                        prompts_path=os.path.join(BASE_DIR, 'chat/prompts.yaml')
                    )
        else:
            raise (f"create_document(): variable methods is not valid: {methods}")
    document = process_document(
        document=document,
        embedding_model=embedding_model,
        methods=methods,
        models=models,
    )
    document = add_uuid_to_document(document)
    return document


def create_document_from_url(filename: list, 
                            config: dict,
                            embedding_model,
                            methods='default',
                            ):
    """
    method: 'texts / summaries'.
    """
    document = read_url_in_document(filename)
    document = add_texts_and_windows_to_document(
        document=document, 
        chunk_length=config['chunk_length'], 
        overlap=config['overlap'],
    )
    if methods=='default':
        llm = None
    else:
        if isinstance(methods, str): 
            methods = list(methods)
        elif isinstance(methods, list):
            models = {}
            for method in methods:
                if method not in models.keys():
                    models[method]= LLM(
                        config=config,
                        chatbot_name=method, 
                        prompts_path='/Users/eugene/Desktop/docstalks/docstalks/chat/prompts.yaml'
                    )
        else:
            raise (f"create_document(): variable methods is not valid: {methods}")
    document = process_document(
        document=document,
        embedding_model=embedding_model,
        methods=methods,
        models=models,
    )
    document = add_uuid_to_document(document)
    return document


def github_connector(url, token):
    pass
    # TODO: from unstructured import github


def stream_text(input):
    for char in input:
        print(char, end='', flush=True)
        delay = round(random.uniform(0.0005, 0.005), 6)
        sleep(delay)


def print_color(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m'
    }
    reset_color = '\033[0m'
    if not isinstance(text, str): 
        text = str(text)
    if color.lower() in colors:
        print(colors[color.lower()] + text + reset_color)
    else:
        print("Invalid color. Available colors are:", ", ".join(colors.keys()))


######################################
### CONNECTOR FOR THE WEB SCRAPING ###
######################################

from bs4 import BeautifulSoup
import requests
from unstructured.partition.html import partition_html


def get_links(url: str, verify: bool, headers: dict) -> list:
    response = requests.get(url, verify=verify, headers=headers)
    data = response.text
    soup = BeautifulSoup(data, 'lxml')
    links = []
    for link in soup.find_all('a'):
        link_url = link.get('href')
        if link_url is not None and link_url.startswith('http'):
            links.append(link_url)
    links.append(url)
    return links


def collect_text_from_partition_html(documents: list):
    if len(documents) > 1:
        docs = [doc.text for doc in documents]
        return '\n'.join(docs)
    else:
        try:
            return documents[0].text
        except:
            return ''


def split_webpage_into_documents(url: str, recursive: bool = False, ssl_verify: bool = False):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
    }
    if recursive:
        links = get_links(url=url, verify=ssl_verify, headers=headers)
        links = list(set(links))  #deleting duplicated links
        elements = dict()
        for link in links:
            docs = partition_html(url=link, ssl_verify=ssl_verify, headers=headers)
            if link not in elements.keys():
                elements[link] = docs
            else:
                elements[link].extend(docs)
        return elements
    else:
        elements = partition_html(url=url, ssl_verify=ssl_verify, headers=headers)
        return {url: elements}


if __name__=="__main__":
    base_url = "https://lenalondonphoto.com/"

    t = time()
    res_default = split_webpage_into_documents(base_url, ssl_verify=False)
    print(time() - t, ' s')

    t = time()
    res_recursive = split_webpage_into_documents(base_url, recursive=True, ssl_verify=False)
    print(time() - t, ' s')
