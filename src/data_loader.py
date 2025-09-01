from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import pdfplumber
import os
from . import config

def load_pdf_documents() -> List[Document]:
    """
    지정된 디렉토리에서 PDF 문서를 로드하고 텍스트를 추출합니다.
    """
    documents = []
    for root, _, files in os.walk(config.DATA_DIR):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                documents.append(Document(page_content=text, metadata={"source": pdf_path}))
    return documents

def load_documents() -> List[Document]:
    """
    지정된 디렉토리에서 .txt 및 .pdf 문서를 로드합니다.
    """
    txt_loader = DirectoryLoader(
        str(config.DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True
    )
    txt_documents = txt_loader.load()
    
    pdf_documents = load_pdf_documents()
    
    return txt_documents + pdf_documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    로드된 문서를 설정에 따라 청크로 분할합니다.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)
