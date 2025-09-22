# Repository Guidelines

## Project Structure & Module Organization
- `app.py`는 Streamlit 기반 대화형 UI로, `src/` 모듈을 조합해 RAG 체인을 초기화합니다.
- `evaluate.py`는 `eval_dataset.csv`를 활용한 RAGAS 평가 스크립트이니 관련 유틸을 함께 유지하세요.
- `src/` 폴더에는 `config.py`(경로 및 모델 ID), `data_loader.py`(PDF/TXT 적재와 분할), `vector_store.py`, `llm.py`가 있으며 새 파이프라인 코드는 여기에 배치한 뒤 `app.py`에서 import합니다.
- `data/raw/`에는 원본 법령·참고 문서를 저장하고, 중간 산출물(FAISS 인덱스, 평가 CSV)은 `data/processed/` 또는 `.gitignore`된 폴더에 보관합니다.
- `requirements.txt`는 필수 파이썬 패키지 목록이므로 추가 의존성도 이 파일과 README의 conda 환경 설명에 맞춰 정리합니다.

## Build, Test, and Development Commands
- `conda create -n legal-rag python=3.13 && conda activate legal-rag`으로 표준 실행 환경을 맞춥니다.
- `pip install -r requirements.txt`로 LangChain, FAISS, Streamlit 스택을 설치합니다.
- `streamlit run app.py`는 로컬 UI를 실행하며 `data/raw/`와 `.env` 값을 사용합니다.
- `python evaluate.py`는 RAGAS 평가를 수행하고 `evaluation_result_baseline.csv`를 생성합니다.
- `python test_parser.py`는 PDF 테이블 파싱을 점검하니 카테고리 누락 메시지를 꼭 확인합니다.

## Coding Style & Naming Conventions
- 파이썬 기본 가이드를 따르고 들여쓰기는 스페이스 4칸, 최대 줄 길이는 100자로 유지합니다.
- 함수와 변수는 snake_case, 상수는 UPPER_CASE를 사용하며, 한국어 UI 문구는 한글 그대로 유지합니다.
- 공개 함수에는 타입 힌트를 권장하고, 경로/모델 설정은 `src/config.py`에서 일관되게 관리합니다.
- 커밋 전 `python -m black app.py evaluate.py src`를 실행해 서식과 따옴표를 정리합니다.

## Testing Guidelines
- `data_loader.py`나 파싱 로직을 수정했다면 `python test_parser.py`를 즉시 실행합니다.
- 앞으로 추가할 `tests/` 패키지에는 `[별표 31]` 테이블과 일반 PDF ingest 회귀 사례를 중심으로 pytest 케이스를 마련합니다.
- `python evaluate.py` 재실행으로 새로운 기준 성능을 기록하고, PR 설명에 지표 변화량을 첨부합니다.

## Commit & Pull Request Guidelines
- 커밋은 작고 명확한 단위로 묶고, 예시처럼 `파싱테스트 추가, 테이블데이터 처리 고도화 완료` 등 한국어 요약을 사용합니다.
- 필요 시 첫 줄에서 관련 이슈를 참조하고 72자 이내로 줄 바꿈합니다.
- PR에는 문제 정의, 해결 요약, 수행한 테스트/평가 명령, 추가 데이터나 시크릿 요구사항, UI 변경 시 Streamlit 스크린샷을 포함합니다.

## Configuration & Secrets
- `.env`에는 `GOOGLE_API_KEY`와 선택적 `EMBED_MODEL`을 설정하되 절대 커밋하지 않습니다.
- 경로가 바뀌면 `src/config.py`를 즉시 갱신하고, 새 디렉터리는 이 문서에 기록합니다.
