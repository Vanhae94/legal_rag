from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from . import config

def load_documents() -> List[dict]:
    """
    지정된 디렉토리에서 문서를 로드합니다.
    .txt 파일만 로드하도록 설정합니다.
    """
    loader = DirectoryLoader(
        str(config.DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True
    )
    return loader.load()

def split_documents(documents: List[dict]) -> List[dict]:
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