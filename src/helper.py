# d:\-Medical-Chatbot\src\helper.py

import os
import glob
from typing import List
from langchain.schema import Document

# FIX: Use langchain_community for all deprecated imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf_file(data):
    # This code is fine, provided imports are fixed
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

# You must also include the filter_to_minimal_docs function here:
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Filter documents to retain only page_content and 'source' metadata."""
    return [
        Document(
            page_content=doc.page_content, 
            metadata={"source": doc.metadata.get("source")}
        )
        for doc in docs
    ]

# And the text_split function:
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

# And the embeddings function with the corrected import:
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 
    return embeddings