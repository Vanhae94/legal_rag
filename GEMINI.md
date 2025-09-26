# GEMINI.md (한국어)

## 프로젝트 개요

이 프로젝트는 검색 증강 생성(RAG) 파이프라인을 구현한 한국 법률 리서치 어시스턴트 챗봇입니다. 사용자가 법률 문서에 대해 질문하면, 제공된 소스 자료를 기반으로 시스템이 답변을 생성하는 웹 인터페이스를 제공합니다.

**주요 기술:**
*   **애플리케이션 프레임워크:** Streamlit
*   **오케스트레이션:** LangChain (LCEL - LangChain Expression Language 사용)
*   **LLM:** Google Gemini (`gemini-1.5-flash-latest`)
*   **임베딩:** HuggingFace Sentence Transformers (`jhgan/ko-sbert-nli`)
*   **벡터 스토어:** FAISS (인메모리)
*   **평가:** Ragas

**아키텍처:**
애플리케이션은 `data/raw` 디렉토리에서 문서(`.txt`, `.pdf`)를 로드하여 청크로 분할하고, 문장 임베딩을 사용해 FAISS 벡터 스토어에 인덱싱합니다. 사용자가 질문을 하면, 시스템은 벡터 스토어에서 관련 문서 청크를 검색하고, 이를 사용자의 쿼리와 결합하여 프롬프트를 만든 후 Gemini LLM으로 전송하여 인용 정보가 포함된 간결한 답변을 생성합니다. 또한, 이 프로젝트는 `ragas` 프레임워크를 사용하여 답변의 신뢰성(faithfulness) 및 관련성(relevancy)과 같은 성능 지표를 측정하는 포괄적인 평가 스크립트를 포함합니다.

## 빌드 및 실행 방법

### 1. 설정

**의존성 설치:**
필요한 Python 패키지를 설치합니다.
```bash
pip install -r requirements.txt
```

**환경 변수:**
프로젝트 루트 디렉토리에 `.env` 파일을 생성합니다. 이 파일에는 Google API 키가 포함되어야 합니다.
```
# .env
GOOGLE_API_KEY="your_google_api_key_here"
```

### 2. 데이터 준비

소스 법률 문서(`.txt` 또는 `.pdf` 형식)를 `data/raw/` 디렉토리에 넣습니다.

또한, 이 프로젝트에는 HWP 파일(`.hwp`)을 텍스트로 변환하는 스크립트가 포함되어 있습니다. HWP 파일이 있는 경우, 프로젝트 루트에 위치시킨 후 다음을 실행하세요:
```bash
python hwp_to_txt.py
```
이 명령은 파일을 변환하여 생성된 `.txt` 파일을 `data/raw/` 디렉토리에 저장합니다.

### 3. 애플리케이션 실행

Streamlit 웹 인터페이스를 시작하려면 터미널에서 다음 명령을 실행하세요:
```bash
streamlit run app.py
```
이후 웹 브라우저에서 챗봇에 접속할 수 있습니다.

### 4. 평가 실행

이 프로젝트는 `ragas`를 사용하여 RAG 파이프라인의 성능을 평가합니다. 평가 데이터는 `eval_dataset.csv`에 정의되어 있습니다.

평가를 실행하려면 다음을 실행하세요:
```bash
# 기본 실험 이름 'baseline'으로 실행
python evaluate.py

# 사용자 지정 실험 이름으로 실행
python evaluate.py my_experiment
```
결과는 `evaluation_result_<experiment_name>.csv`라는 이름의 CSV 파일로 저장됩니다.

## 개발 컨벤션

*   **코드 구조:** 핵심 로직은 `src/` 디렉토리 내에서 모듈화되어 있습니다:
    *   `config.py`: 경로 및 모델 매개변수와 같은 설정을 관리합니다.
    *   `data_loader.py`: 복잡한 PDF 테이블 파싱을 포함하여 소스 문서의 로딩 및 처리를 담당합니다.
    *   `llm.py`: Gemini LLM 인스턴스를 구성하고 제공합니다.
    *   `vector_store.py`: 임베딩 모델 및 FAISS 벡터 스토어 생성을 관리합니다.
*   **메인 애플리케이션:** `app.py`는 Streamlit UI와 메인 RAG 체인 오케스트레이션을 포함합니다.
*   **RAG 체인:** RAG 체인은 LangChain Expression Language(LCEL)를 사용하여 구성되어 데이터 흐름을 명시적이고 수정하기 쉽게 만듭니다.
*   **캐싱:** Streamlit의 `@st.cache_resource`를 사용하여 RAG 체인을 캐시하여 모든 사용자 상호작용 시 재초기화를 방지하고 성능을 향상시킵니다.
*   **평가:** `evaluate.py` 스크립트는 RAG 시스템의 품질을 측정하고 추적하는 표준화된 방법을 제공하여 반복적인 개선을 용이하게 합니다.