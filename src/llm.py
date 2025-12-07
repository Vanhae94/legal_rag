import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    """
    .env 파일에서 API 키를 로드하고, Google Gemini 모델 인스턴스를 반환합니다.
    """
    # .env 파일에서 환경 변수를 로드합니다.
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")

    # Google Gemini 모델을 초기화합니다.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0,
    )
    
    return llm