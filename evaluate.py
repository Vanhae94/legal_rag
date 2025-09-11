import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
import asyncio
from dotenv import load_dotenv

from app import get_rag_chain_with_source
from src.llm import get_llm
from src.config import EMBED_MODEL
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# .env 파일 로드
load_dotenv()

# asyncio 이벤트 루프 관련 설정 (Windows 환경에서 필요할 수 있음)
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def create_evaluation_dataset() -> Dataset:
    """
    CSV 파일에서 평가 데이터셋을 로드하고, RAG 체인을 실행하여
    'answer'와 'contexts'를 추가한 뒤 Hugging Face Dataset으로 변환합니다.
    """
    # 1. 평가 데이터셋 로드
    eval_df = pd.read_csv("eval_dataset.csv")
    questions = eval_df["question"].tolist()
    ground_truths = eval_df["ground_truth"].tolist()

    # 2. RAG 체인 로드
    print("RAG 체인을 로드하는 중입니다...")
    rag_chain = get_rag_chain_with_source()
    if rag_chain is None:
        raise ValueError("RAG 체인을 로드할 수 없습니다. 데이터 파일이 있는지 확인하세요.")
    print("RAG 체인 로드 완료.")

    # 3. 각 질문에 대해 RAG 체인 실행 및 결과 수집
    answers = []
    contexts = []
    print(f"{len(questions)}개의 질문에 대해 답변 및 근거 문서를 생성합니다...")
    for query in questions:
        try:
            result = rag_chain.invoke(query)
            answers.append(result["answer"])
            contexts.append([doc.page_content for doc in result["context"]])
        except Exception as e:
            print(f"질문 '{query}' 처리 중 오류 발생: {e}")
            answers.append("")
            contexts.append([])
    
    print("답변 생성 완료.")

    # 4. RAGAs가 요구하는 형식으로 데이터 구성
    response_dataset = Dataset.from_dict({
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts,
    })

    return response_dataset

def run_evaluation(dataset: Dataset):
    """
    주어진 데이터셋에 대해 RAGAs 평가를 실행하고 결과를 출력합니다.
    """
    print("RAGAs 평가를 시작합니다...")
    
    # Ragas에서 사용할 모델 초기화
    ragas_llm = LangchainLLMWrapper(get_llm())
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )

    # 평가 지표 정의
    metrics = [
        faithfulness,       # 답변이 근거에 충실한가
        answer_relevancy,   # 답변이 질문과 관련 있는가
        context_precision,  # 검색된 컨텍스트가 질문과 관련 있는가
        context_recall,     # 답변에 필요한 컨텍스트를 모두 검색했는가
    ]

    # 평가 실행
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    
    print("RAGAs 평가 완료.")
    
    # 결과 출력
    print(result)
    return result

if __name__ == "__main__":
    # 평가 데이터셋 생성
    evaluation_dataset = create_evaluation_dataset()
    
    # 평가 실행
    evaluation_result = run_evaluation(evaluation_dataset)
    
    # 결과를 DataFrame으로 변환하여 CSV 파일로 저장
    df_result = evaluation_result.to_pandas()
    df_result.to_csv("evaluation_result_baseline.csv", index=False, encoding="utf-8-sig")
    print("\n'evaluation_result_baseline.csv' 파일로 초기 성능 평가 결과가 저장되었습니다.")