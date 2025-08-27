from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from . import config

def get_embedding_model():
    """임베딩 모델을 로드합니다."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def build_vector_store(documents: List[dict], embeddings) -> FAISS:
    """
    문서 청크로부터 FAISS 벡터 저장소를 구축합니다.
    """
    print("빌드 시작")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("빌드 완료")
    return vector_store