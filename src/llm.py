import httpx
from typing import List, Dict, Any, Tuple

from . import config

def to_context(retrieved: List[Tuple[float, Dict[str, Any]]]) -> str:
    """검색된 결과를 LLM에 전달할 컨텍스트 문자열로 변환합니다."""
    blocks = []
    for s, rec in retrieved:
        m = rec["meta"]
        title = m.get("title", "")
        src = m.get("source", m.get("law", ""))
        blocks.append(f"[TITLE] {title}\n[SOURCE] {src}\n[CHUNK] {rec['text']}\n---")
    return "\n".join(blocks)

def call_llm(messages: List[Dict[str, str]], temperature=0.2, max_tokens=800) -> str:
    """OpenRouter 또는 Ollama API를 호출하여 LLM 응답을 받습니다."""
    # OpenRouter 우선, 없으면 Ollama
    if config.OPENROUTER_API_KEY:
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "HTTP-Referer": config.APP_URL,
            "X-Title": config.APP_NAME,
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.OPENROUTER_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            with httpx.Client(timeout=60) as c:
                r = c.post("https://openrouter.ai/api/v1/chat/completions",
                           json=payload, headers=headers)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            print(f"OpenRouter API 오류: {e}")
            # OpenRouter 실패 시 Ollama로 넘어가지 않고 에러를 알림
            return "API 호출에 실패했습니다. API 키와 모델 설정을 확인해주세요."
    else:
        url = f"{config.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": False, # stream False
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        try:
            with httpx.Client(timeout=120) as c:
                r = c.post(url, json=payload)
                r.raise_for_status()
                return r.json().get("message", {}).get("content", "")
        except httpx.RequestError as e:
            print(f"Ollama 연결 오류: {e}")
            return "Ollama 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요."
