from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import pdfplumber
import os
from . import config

def load_pdf_documents() -> List[Document]:
    """
    지정된 디렉토리에서 PDF 문서를 로드하고, 페이지별 텍스트와 테이블을 별도의 Document 객체로 생성합니다.
    테이블은 Markdown 형식으로 변환됩니다.
    """
    documents = []
    for root, _, files in os.walk(config.DATA_DIR):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        # 1. 페이지의 일반 텍스트를 별도의 Document로 추가
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            documents.append(Document(
                                page_content=page_text,
                                metadata={"source": pdf_path, "page": i + 1}
                            ))

                        # 2. 페이지의 각 테이블을 별도의 Document로 추가
                        for table in page.extract_tables():
                            if table:
                                # None 값을 빈 문자열로 처리하여 안정성 확보
                                header_list = [str(h) if h is not None else "" for h in table[0]]
                                
                                # Markdown 테이블 생성
                                header = "| " + " | ".join(header_list) + " |"
                                separator = "| " + " | ".join(["---"] * len(header_list)) + " |"
                                
                                body_rows = []
                                for row in table[1:]:
                                    row_list = [str(c) if c is not None else "" for c in row]
                                    # 행의 셀 개수를 헤더 개수에 맞춤
                                    row_list = (row_list + [''] * len(header_list))[:len(header_list)]
                                    body_rows.append("| " + " | ".join(row_list) + " |")
                                body = "\n".join(body_rows)
                                
                                markdown_table = f"{header}\n{separator}\n{body}"
                                
                                documents.append(Document(
                                    page_content=markdown_table,
                                    metadata={"source": pdf_path, "page": i + 1, "type": "table"}
                                ))
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