from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import pdfplumber
import os
from . import config

def load_pdf_documents() -> List[Document]:
    """
    지정된 디렉토리에서 PDF 문서를 로드하고, 페이지별 텍스트와 테이블을 처리합니다.
    - 일반 텍스트는 페이지별로 Document 객체를 생성합니다.
    - 테이블은 각 행(row)을 의미 있는 자연어 문장으로 변환하여 별도의 Document 객체로 생성합니다.
    """
    documents = []
    for root, _, files in os.walk(config.DATA_DIR):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                file_name = os.path.basename(pdf_path)  # 파일명을 표의 제목처럼 활용
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        for i, page in enumerate(pdf.pages):
                            # 1. 페이지의 일반 텍스트를 별도의 Document로 추가
                            page_text = page.extract_text() or ""
                            if page_text.strip():
                                documents.append(Document(
                                    page_content=page_text,
                                    metadata={"source": file_name, "page": i + 1, "type": "text"}
                                ))

                            # 2. 페이지의 각 테이블을 행 단위 문장으로 변환하여 Document로 추가
                            tables = page.extract_tables()
                            if not tables:
                                continue

                            for table in tables:
                                if not table or len(table) < 2:  # 헤더와 최소 1개 행이 있는지 확인
                                    continue

                                # None 값을 빈 문자열로 변환
                                header = [str(h).strip() if h is not None else "" for h in table[0]]
                                
                                # 유효한 헤더가 있는지 확인
                                if not any(header):
                                    continue

                                body_rows = table[1:]

                                for row_data in body_rows:
                                    # 행 데이터도 None 값을 빈 문자열로 변환
                                    row = [str(c).strip() if c is not None else "" for c in row_data]
                                    
                                    # 행의 셀 개수를 헤더 개수에 맞춤
                                    if len(row) != len(header):
                                        continue  # 헤더와 행의 길이가 다르면 일단 건너뜀 (오류 방지)

                                    # 문장 생성 로직
                                    key_parts = []
                                    value_parts = []
                                    
                                    # 기준이 되는 키 컬럼 개수 (보통 1~2개)
                                    num_key_cols = min(2, len(header) - 1) if len(header) > 1 else 1

                                    for idx, (h, c) in enumerate(zip(header, row)):
                                        if not h or not c:  # 헤더나 셀이 비어있으면 무시
                                            continue
                                        # 개행 문자를 공백으로 치환하여 문장의 흐름을 유지
                                        c = c.replace('\n', ' ')
                                        if idx < num_key_cols:
                                            key_parts.append(f"'{h}'이(가) '{c}'")
                                        else:
                                            value_parts.append(f"{h}: {c}")
                                    
                                    if not key_parts or not value_parts:
                                        continue

                                    key_str = "이고 ".join(key_parts)
                                    value_str = ", ".join(value_parts)
                                    
                                    sentence = f"'{file_name}' 문서의 표에서 {key_str}인 경우, 세부 내용은 다음과 같습니다: {value_str}."
                                    
                                    documents.append(Document(
                                        page_content=sentence,
                                        metadata={"source": file_name, "page": i + 1, "type": "table_row"}
                                    ))
                except Exception as e:
                    print(f"'{pdf_path}' 파일 처리 중 오류 발생: {e}")  # 오류 로깅 추가
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