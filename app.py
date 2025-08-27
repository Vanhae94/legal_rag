from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src import config, data_loader, vector_store, llm

def format_docs(docs):
    """검색된 문서를 프롬프트에 삽입할 문자열로 포맷합니다."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    """LangChain 기반 RAG 애플리케이션 메인 함수"""
    # 1. 데이터 로드 및 분할
    print("문서를 로드하고 분할합니다...")
    docs = data_loader.load_documents()
    if not docs:
        print(f"⚠️  데이터 파일을 찾을 수 없습니다. '{config.DATA_DIR}' 디렉토리에 .txt 파일을 넣어주세요.")
        return
    splits = data_loader.split_documents(docs)

    # 2. 임베딩 및 벡터 저장소 생성
    print("임베딩 모델을 로드하고 벡터 저장소를 구축합니다...")
    embeddings = vector_store.get_embedding_model()
    db = vector_store.build_vector_store(splits, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 6})

    # 3. LLM 인스턴스화
    llm_instance = llm.get_llm()

    # 4. 프롬프트 템플릿 정의
    template = """
<컨텍스트>
{context}

[질문]
{question}

형식:
- 요약(2~5문장)
- 핵심포인트(불릿)
- 인용(제목/출처)
- 면책고지

최종 답만 한국어로. 사고흐름/메타 금지.
"""
    prompt = PromptTemplate.from_template(template)

    # 5. LCEL을 사용한 RAG 체인 구성
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_instance
        | StrOutputParser()
    )

    print("\n✅ RAG 체인 준비 완료. 질문을 입력하세요.")

    # 6. 사용자 질문 입력 루프
    while True:
        query = input("\n질문 (종료하려면 'exit' 입력): ").strip()
        if not query:
            continue
        if query.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break

        # 7. 체인 실행 및 결과 출력
        answer = rag_chain.invoke(query)
        print("\n=== 답변 ===\n", answer.strip())

if __name__ == "__main__":
    main()
