import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Embedding Model
EMBED_MODEL = os.getenv("EMBED_MODEL", "jhgan/ko-sbert-nli")

# LLM API (OpenRouter / Ollama)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# App Info
APP_URL  = os.getenv("APP_URL", "http://localhost")
APP_NAME = os.getenv("APP_NAME", "LegalRAG")

# Data/Chunking
DATA_DIR = Path("data/raw")
CHUNK_SIZE, OVERLAP = 400, 60

# System Prompt
SYSTEM_PROMPT = (
"당신은 한국 법령 리서처입니다. 아래 <컨텍스트>만을 근거로 "
"간결하고 정확히 답변하고, 인용(제목/출처)을 제시하세요. "
"모르면 모른다고 말하세요. 마지막에 '법률 자문 아님' 고지를 넣으세요."
)
