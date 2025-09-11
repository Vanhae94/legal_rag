from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import pdfplumber
import os
import re
from . import config

def parse_complex_table(table: list, file_name: str, page_num: int, category: str) -> List[Document]:
    """
    (최종 수정 로직) pdfplumber가 추출한 원본 테이블 구조에 맞춰, 
    다차원 체력검정 기준표를 정확하게 파싱하여 문장을 생성합니다.
    'category'를 인자로 받아 문장에 포함시키고, 텍스트 정제 및 시간 형식 변환을 수행합니다.
    """
    documents = []
    if not table or len(table) < 3:
        return documents

    def clean_text(text: str) -> str:
        """텍스트 내 불필요한 공백과 개행문자를 제거합니다."""
        if not text:
            return ""
        return text.replace('\n', '').replace(' ', '')

    def format_time(text: str) -> str:
        """'18:11' 같은 시간 형식을 '18분 11초'로 변환합니다."""
        # M:S-M:S 범위 형식 처리
        text = re.sub(r'(\d+):(\d+)-(\d+):(\d+)', r'\1분 \2초 - \3분 \4초', text)
        # M:S 형식 처리
        text = re.sub(r'(\d+):(\d+)', r'\1분 \2초', text)
        return text

    age_header_row = table[1]
    age_map = {i: age for i, age in enumerate(age_header_row) if age and age.strip()}

    data_rows = table[2:]
    current_sport = ""
    current_pass_fail = ""

    for row in data_rows:
        if row[0] and row[0].strip():
            current_sport = clean_text(row[0])
        
        if row[1] and row[1].strip():
            current_pass_fail = clean_text(row[1])

        grade_raw = row[2] if row[2] and row[2].strip() else ""
        if not grade_raw:
            continue
        grade = clean_text(grade_raw)

        for c_idx, cell_value_raw in enumerate(row):
            if c_idx < 3 or not cell_value_raw or not cell_value_raw.strip():
                continue

            age_group = age_map.get(c_idx)
            if not age_group:
                continue
            
            # 데이터 정제
            cleaned_cell_value = clean_text(cell_value_raw)
            formatted_cell_value = format_time(cleaned_cell_value)

            context_parts = [f"'구분': '{category}'"]
            if current_sport:
                context_parts.append(f"'종목': '{current_sport}'")
            if current_pass_fail and current_pass_fail == '불합격':
                 context_parts.append(f"'합격여부': '{current_pass_fail}'")
            if grade:
                context_parts.append(f"'등급': '{grade}'")
            
            context_parts.append(f"'나이': '{clean_text(age_group)}'")
            context_str = ", ".join(context_parts)
            
            sentence = (
                f"'{file_name}' 문서의 체력검정 기준표에 따르면, "
                f"{context_str} 조건의 기준은 '{formatted_cell_value}'입니다."
            )
            documents.append(Document(
                page_content=sentence,
                metadata={"source": file_name, "page": page_num, "type": "complex_table_row", "category": category}
            ))
            
    return documents

def load_pdf_documents() -> List[Document]:
    """
    지정된 디렉토리에서 PDF 문서를 로드하고, 페이지별 텍스트와 테이블을 처리합니다.
    - '[별표 31]' 파일은 카테고리(남군, 여군 등)를 식별하여 특별 파싱합니다.
    - 그 외 PDF는 일반 텍스트와 간단한 테이블 행을 파싱합니다.
    """
    documents = []
    for root, _, files in os.walk(config.DATA_DIR):
        for file in files:
            if not file.endswith(".pdf"):
                continue
            
            pdf_path = os.path.join(root, file)
            file_name = os.path.basename(pdf_path)

            # '[별표 31]' 파일은 특별 로직으로 처리
            if file.startswith("[별표 31]"):
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        all_categories = []
                        all_tables_with_pages = []

                        # 1. PDF의 모든 페이지를 순회하며 카테고리와 테이블을 순서대로 수집
                        for i, page in enumerate(pdf.pages):
                            page_num = i + 1
                            page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                            
                            # 카테고리 찾기 (정규식에서 $ 제거)
                            found_categories = re.findall(r'^\d+\.\s*(남\s*군|여\s*군|남자\s*군무원|여자\s*군무원)', page_text, re.MULTILINE)
                            if found_categories:
                                all_categories.extend([c.replace(" ", "") for c in found_categories])
                            
                            # 테이블 찾기
                            tables = page.extract_tables()
                            if tables:
                                for table in tables:
                                    all_tables_with_pages.append({"page": page_num, "table": table})
                            
                            # 원본 페이지 텍스트는 항상 추가
                            if page_text.strip():
                                documents.append(Document(
                                    page_content=page_text,
                                    metadata={"source": file_name, "page": page_num, "type": "text"}
                                ))

                        # 2. 수집된 카테고리와 테이블을 1:1로 매칭하여 파싱
                        if len(all_categories) == len(all_tables_with_pages):
                            for i, table_info in enumerate(all_tables_with_pages):
                                category = all_categories[i]
                                table = table_info["table"]
                                page_num = table_info["page"]
                                documents.extend(parse_complex_table(table, file_name, page_num, category))
                        else:
                            print(f"Warning in {file_name}: Found {len(all_categories)} categories and {len(all_tables_with_pages)} tables. Could not perform reliable parsing.")

                except Exception as e:
                    print(f"'{pdf_path}' 파일 처리 중 오류 발생: {e}")
                continue

            # --- 그 외 모든 PDF 파일에 대한 일반 처리 ---
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        page_num = i + 1
                        page_text = page.extract_text() or ""
                        
                        if page_text.strip():
                            documents.append(Document(
                                page_content=page_text,
                                metadata={"source": file_name, "page": page_num, "type": "text"}
                            ))

                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                if not table or len(table) < 2:
                                    continue
                                header = [str(h).strip() if h is not None else "" for h in table[0]]
                                if not any(header):
                                    continue
                                body_rows = table[1:]
                                for row_data in body_rows:
                                    row = [str(c).strip() if c is not None else "" for c in row_data]
                                    if len(row) != len(header):
                                        continue
                                    key_parts, value_parts = [], []
                                    num_key_cols = min(2, len(header) - 1) if len(header) > 1 else 1
                                    for idx, (h, c) in enumerate(zip(header, row)):
                                        if not h or not c: continue
                                        c = c.replace('\n', ' ')
                                        if idx < num_key_cols:
                                            key_parts.append(f"'{h}'이(가) '{c}'")
                                        else:
                                            value_parts.append(f"{h}: {c}")
                                    if not key_parts or not value_parts: continue
                                    key_str = "이고 ".join(key_parts)
                                    value_str = ", ".join(value_parts)
                                    sentence = f"'{file_name}' 문서의 표에서 {key_str}인 경우, 세부 내용은 다음과 같습니다: {value_str}."
                                    documents.append(Document(
                                        page_content=sentence,
                                        metadata={"source": file_name, "page": page_num, "type": "table_row"}
                                    ))
            except Exception as e:
                print(f"'{pdf_path}' 파일 처리 중 오류 발생: {e}")
    return documents

def load_txt_documents() -> List[Document]:
    """
    지정된 디렉토리에서 .txt 문서를 로드합니다.
    """
    txt_loader = DirectoryLoader(
        str(config.DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True
    )
    return txt_loader.load()

def load_all_documents() -> List[Document]:
    """
    지정된 디렉토리에서 모든 문서(.txt, .pdf)를 로드하고 결합합니다.
    """
    txt_documents = load_txt_documents()
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