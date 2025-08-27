import httpx
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LLM

from . import config

class CustomOllama(LLM):
    """Ollama 모델을 위한 사용자 정의 LangChain LLM 래퍼"""
    
    @property
    def _llm_type(self) -> str:
        return "custom_ollama"

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> str:
        url = f"{config.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": kwargs.get('temperature', 0.2), "num_predict": kwargs.get('max_tokens', 800)}
        }
        try:
            with httpx.Client(timeout=120) as c:
                r = c.post(url, json=payload)
                r.raise_for_status()
                return r.json().get("response", "")
        except httpx.RequestError as e:
            return f"Ollama 연결 오류: {e}"
        except httpx.HTTPStatusError as e:
            return f"Ollama API 오류: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": config.OLLAMA_MODEL}

class CustomOpenRouter(LLM):
    """OpenRouter 모델을 위한 사용자 정의 LangChain LLM 래퍼"""

    @property
    def _llm_type(self) -> str:
        return "custom_openrouter"

    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None, 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> str:
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": config.APP_URL,
            "X-Title": config.APP_NAME,
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}], # 단순화를 위해 프롬프트를 user 메시지로 전달
            "temperature": kwargs.get('temperature', 0.2),
            "max_tokens": kwargs.get('max_tokens', 800)
        }
        try:
            with httpx.Client(timeout=60) as c:
                r = c.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"OpenRouter API 오류: {e}"
        except Exception as e:
            return f"알 수 없는 오류: {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": config.OPENROUTER_MODEL}

def get_llm() -> LLM:
    """설정에 따라 적절한 LLM 인스턴스를 반환합니다."""
    if config.OPENROUTER_API_KEY:
        return CustomOpenRouter()
    return CustomOllama()