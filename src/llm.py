import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def get_llm():
    """
    .env 파일에서 API 키를 로드하고, OpenRouter 모델 인스턴스를 반환합니다.
    """
    # .env 파일에서 환경 변수를 로드합니다.
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY가 .env 파일에 설정되지 않았습니다.")

    # OpenRouter 모델을 초기화합니다.
    # ref: https://openrouter.ai/docs#quick-start
    llm = ChatOpenAI(
        model="x-ai/grok-4-fast:free",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8501", # app url
            "X-Title": "Korean Legal RAG", # app name
        }
    )
    
    return llm