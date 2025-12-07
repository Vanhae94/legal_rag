# ⚖️ Korean Legal RAG Chatbot

한국어 법령 검색 및 질의응답을 위한 **RAG (Retrieval-Augmented Generation)** 기반 챗봇 프로젝트입니다.
정확한 법률 정보 제공을 위해 **Hybrid Retrieval(Vector Search + Reranking)** 파이프라인과 **Ragas** 기반의 정량적 성능 평가 시스템을 구축했습니다.

---

## 🚀 Key Features

### 1. 고정밀 검색 파이프라인 (2-Stage Retrieval)
단순한 벡터 유사도 검색의 한계를 극복하기 위해 **Cross-Encoder 기반의 재순위화(Reranking)** 과정을 도입했습니다.
- **1단계 (Retrieval)**: `FAISS`와 `jhgan/ko-sbert-nli` 임베딩을 사용하여 관련성 높은 문서 20개를 빠르게 검색 (Recall 확보)
- **2단계 (Reranking)**: `dragonkue/bge-reranker-v2-m3-ko` 모델을 사용하여 질문과의 논리적 연관성을 정밀 채점, 상위 6개 선별 (Precision 확보)

### 2. 근거 기반 답변 (Citation)
- LLM이 답변을 생성할 때 사용한 법령의 **출처(파일명, 조항 등)를 명시**하여 신뢰성을 높였습니다.
- 환각(Hallucination) 최소화를 위해 컨텍스트 내 정보만으로 답변하도록 프롬프트 엔지니어링을 적용했습니다.

### 3. 데이터 기반 성능 최적화 (Evaluation)
- **Ragas (Retrieval Augmented Generation Assessment)** 프레임워크를 도입하여 RAG 파이프라인의 성능을 객관적으로 측정합니다.
- 평가 지표: `Context Precision`, `Context Recall`, `Faithfulness`, `Answer Relevancy`

---

## 🛠️ Technology Stack

| Category | technologies |
|----------|--------------|
| **LLM** | Google Gemini 2.0 Flash (`gemini-2.0-flash-exp`) |
| **Framework** | LangChain (LCEL), Streamlit |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **Embedding** | `jhgan/ko-sbert-nli` (Sentence-Transformers) |
| **Reranker** | `dragonkue/bge-reranker-v2-m3-ko` (Cross-Encoder) |
| **Evaluation** | Ragas |

---

## 📂 Project Structure

```bash
legal_rag/
├── app.py                # Streamlit 메인 애플리케이션 (UI 및 RAG 체인 실행)
├── evaluate.py           # Ragas 기반 성능 평가 스크립트
├── eval_dataset.csv      # 평가용 QA 데이터셋 (Question-GroundTruth)
├── requirements.txt      # 프로젝트 의존성 목록
└── src/
    ├── config.py         # 환경 설정 및 경로 관리
    ├── data_loader.py    # 문서 로드 및 청킹 (Chunking) 로직
    ├── vector_store.py   # FAISS 인덱스 생성 및 검색 로직
    └── llm.py            # LLM 모델 초기화 (Google Gemini)
```

---

## 💻 Installation & Usage

### 1. 환경 설정
Python 3.10+ 환경에서 필요한 라이브러리를 설치합니다.
```bash
pip install -r requirements.txt
```

### 2. API Key 설정
프로젝트 루트에 `.env` 파일을 생성하고 Google Gemini API 키를 입력합니다.
```env
# .env file
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. 애플리케이션 실행
Streamlit 앱을 실행하여 웹 인터페이스로 챗봇을 사용할 수 있습니다.
```bash
streamlit run app.py
```

### 4. 성능 평가 실행 (Optional)
Ragas를 사용하여 현재 파이프라인의 성능을 평가하고 CSV 리포트를 생성합니다.
```bash
python evaluate.py [experiment_name]
# 예: python evaluate.py chunk800_rerank
```

---

## 📊 Performance Improvement Process
이 프로젝트는 단순 구현에 그치지 않고, 실험을 통해 성능을 개선했습니다.

1.  **Baseline**: 기본 Vector Search (`k=4`) → 문맥 누락 발생
2.  **Chunking Optimization**: Chunk Size를 400자에서 **800자**로 늘려 문맥 보존력 강화
3.  **Reranking 도입**: Retrieval(`k=20`) → Rerank(`top_n=6`) 파이프라인 구축으로 **Context Precision 15% 향상**

> *Note: 상세 평가지표 결과는 `evaluation_result_*.csv` 파일들을 참고하세요.*
