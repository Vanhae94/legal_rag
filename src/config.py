import os

# --- 경로 설정 ---
# 현재 파일의 절대 경로
_current_dir = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트 디렉토리 (src 폴더의 부모)
_project_root = os.path.dirname(_current_dir)

# 데이터 디렉토리
DATA_DIR = os.path.join(_project_root, "data", "raw")

# --- 모델 설정 ---
# HuggingFace 임베딩 모델 이름
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"