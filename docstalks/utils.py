from unstructured.partition.pdf import partition_pdf
import uuid
import time
import random

# from typing import Any
# from pydantic import BaseModel
# from unstructured_inference.models.base import get_model
# from unstructured_inference.inference.layout import DocumentLayout
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# import fitz
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain import hub
# from transformers import AutoTokenizer
# import torch.nn.functional as F
# from torch import Tensor



def read_pdf_in_document(file: str):
    try:
        parts = partition_pdf(
            filename=file,
            include_page_breaks=False,
            strategy='fast',
            infer_table_structure=False,
            include_metadata = True,
            chunking_strategy = None,
        )
        document = parts[0]
        texts = " ".join([part.text.strip() for part in parts])
        document.text = texts
        document.metadata = {
            'file_directory': document.metadata.file_directory,
            'filename': document.metadata.filename,
            'languages': document.metadata.languages,
            'last_modified': document.metadata.last_modified,
            'number_of_pages': parts[-1].metadata.page_number,
            'filetype': document.metadata.filetype,
        }
        return document
    except Exception as e:
        print(f"🛑 Exception called by 'extract_pdf_elemets_with_unstructured' \
        function processing the document: {fname}") 
        print(f"Exception: {e}")


def sumarize_tables(table_elements: list, summarize_chain) -> tuple:
    # Apply to tables
    tables = [i.text for i in table_elements]
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    return tables, table_summaries


def sumarize_texts(text_elements: list, summarize_chain) -> tuple:
    try:
        texts = [i.text for i in text_elements]
    except:
        texts = text_elements[0]
    print(texts)
    # finally:
    #     raise TypeError(f"sumarize_texts function didn't processed the input: {text_elements}\n\
    #     Required input is List[srt,]")
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})    
    return texts, text_summaries


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


# def read_file_with_text_splitter(file_path, text_splitter):
#     seps = get_separators(file_path)
#     text = extract_text_from_pdf(file_path)
#     texts = text_splitter.create_documents(text)
#     return texts


# def split_documents(chunk_size: int, docs: list, tokenizer_name: str) -> list:
#     """
#     Split documents into chunks of maximum size 'chunk_size' tokens and return a list of documents.
#     """
#     separators = get_separators(docs[0])
#     # tokenizer_name = 'thenlper/gte-small' if tokenizer_name == '' else tokenizer_name
#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True),
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_size//8,
#         separators=separators,
#     )
#     docs_processed = []
#     for doc in docs:
#         ext = doc.split('.')[-1]
#         if ext == 'pdf':  
#             red_doc = read_pdf(doc)
#             docs_processed += text_splitter.split_documents(red_doc)
#         else:
#             print(f"WARNING: Type of the file '{doc}' is not supported.")
#             continue
#     unique_texts = {}
#     docs_processed_unique = []
#     for doc in docs_processed:
#         if doc.page_content not in unique_texts:
#             unique_texts[doc.page_content] = True
#             docs_processed_unique.append(doc)
#     return docs_processed_unique


def read_pdf_in_document(file: str):
    try:
        parts = partition_pdf(
            filename=file,
            include_page_breaks=False,
            strategy='fast',
            infer_table_structure=False,
            include_metadata = True,
            chunking_strategy = None,
        )
        document = parts[0]
        merged_text = " ".join([part.text.strip() for part in parts])
        document.text = merged_text
        document.metadata = {
            'file_directory': document.metadata.file_directory,
            'filename': document.metadata.filename,
            'languages': document.metadata.languages,
            'last_modified': document.metadata.last_modified,
            'number_of_pages': parts[-1].metadata.page_number,
            'filetype': document.metadata.filetype,
            'texts': [],
            'windows': [],
            'uuid': [],
        }
        document.category = 'Docstalks'
        return document
    except Exception as e:
        print(f"""🛑 Exception called by 'extract_pdf_elemets_with_unstructured'
              function processing the document: {file}""") 
        print(f"Exception: {e}")


def add_texts_and_windows_to_document(document, chunk_length, overlap):
    text = document.text
    words = text.split()
    start = 0
    window_size = chunk_length * 0.8
    window_start = start + window_size
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


def add_embeddings_to_document(document, embedding_model):
    embeddings = [get_embedding_from_text(t, embedding_model) for t in document.metadata['texts']]
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


def create_document(filename: str, chunk_length: int, embedding_model):
    document = read_pdf_in_document(filename)
    document = add_texts_and_windows_to_document(
        document=document, chunk_length=chunk_length, overlap=chunk_length//10
        )
    document = add_embeddings_to_document(document=document, embedding_model=embedding_model)
    document = add_uuid_to_document(document)
    return document


def stream_text(input):
    for char in input:
        print(char, end='', flush=True)
        delay = round(random.uniform(0.0005, 0.005), 6)
        time.sleep(delay)