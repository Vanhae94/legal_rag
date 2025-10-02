from sentence_transformers import CrossEncoder

# 다른 공개된 cross-encoder 모델 테스트
model = CrossEncoder("dragonkue/bge-reranker-v2-m3-ko")
#model = CrossEncoder("bongsoo/korean-cross-encoder-v1")
print("모델 로드 성공!")