# from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
# from unstructured_inference.models.base import get_model
# from unstructured_inference.inference.layout import DocumentLayout
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# import fitz
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain import hub


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


def read_file_with_text_splitter(file_path, text_splitter):
    seps = get_separators(file_path)
    text = extract_text_from_pdf(file_path)
    texts = text_splitter.create_documents(text)
    return texts


def split_documents(chunk_size: int, docs: list, tokenizer_name: str) -> list:
    """
    Split documents into chunks of maximum size 'chunk_size' tokens and return a list of documents.
    """
    separators = get_separators(docs[0])
    # tokenizer_name = 'thenlper/gte-small' if tokenizer_name == '' else tokenizer_name
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True),
        chunk_size=chunk_size,
        chunk_overlap=chunk_size//8,
        separators=separators,
    )
    docs_processed = []
    for doc in docs:
        ext = doc.split('.')[-1]
        if ext == 'pdf':  
            red_doc = read_pdf(doc)
            docs_processed += text_splitter.split_documents(red_doc)
        else:
            print(f"WARNING: Type of the file '{doc}' is not supported.")
            continue
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    return docs_processed_unique


from transformers import AutoTokenizer
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import SentenceTransformer, util


tokenizer_name = 'intfloat/e5-base-v2'
embedding_model = SentenceTransformer(tokenizer_name)


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


def split_text_by_chunks(document, chunk_length, overlap):
    text = document.text
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_length, len(words))
        chunk = " ".join(words[int(start):int(end)])  # Join words to form text chunk
        # text_window_for_the_chunk = 
        chunks.append(chunk)  # Append tuple of (text chunk, tokenized chunk)
        start += chunk_length - overlap
    document.text = chunks
    return document


def add_embeddings_to_document(document, embedding_model):

    # TODO: add embeddings for each of chunk in the docuent.text chunks!
    texts = document.text
    
    embedding = embedding_model.encode(doc.text, convert_to_tensor=True)
    
    return embedding.tolist()