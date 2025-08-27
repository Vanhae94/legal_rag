import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

from . import config
from . import data_loader

# 한 번 로드된 모델을 재사용하기 위한 캐시
_model = None

def get_embedding_model():
    """임베딩 모델을 로드하고 캐시합니다."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model

def build_index(docs: List[Dict[str, Any]]) -> Tuple[faiss.Index, List[str], List[Dict[str, Any]]]:
    """문서 목록에서 Faiss 인덱스를 구축합니다."""
    model = get_embedding_model()
    chunks, metas = [], []
    for d in docs:
        for i, c in enumerate(data_loader.chunk(d["text"])):
            chunks.append(c)
            metas.append({"title": d["title"], **d.get("meta", {})})
    
    vecs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, chunks, metas

def retrieve(index: faiss.Index, chunks: List[str], metas: List[Dict[str, Any]], query: str, k=6) -> List[Tuple[float, Dict[str, Any]]]:
    """쿼리를 사용하여 인덱스에서 문서를 검색합니다."""
    model = get_embedding_model()
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(qv, k)
    
    out = []
    for s, i in zip(scores[0], idxs[0]):
        out.append((float(s), {"text": chunks[i], "meta": metas[i]}))
    return out
