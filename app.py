from src import config, data_loader, vector_store, llm

def main():
    """메인 RAG 애플리케이션 함수"""
    # 1. 데이터 로드
    docs = data_loader.read_docs()
    if not docs:
        print(f"⚠️  데이터 파일을 찾을 수 없습니다. '{config.DATA_DIR}' 디렉토리에 .txt 파일을 넣어주세요.")
        return

    # 2. 벡터 인덱스 빌드 (메모리 사용)
    print("문서 목록을 읽어 벡터 인덱스를 구축합니다...")
    index, chunks, metas = vector_store.build_index(docs)
    print("인덱스 구축 완료.")

    # 3. 사용자 질문 입력
    while True:
        query = input("\n질문을 입력하세요 (종료하려면 'exit' 입력): ").strip()
        if not query:
            continue
        if query.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break

        # 4. 문서 검색 (Retrieve)
        hits = vector_store.retrieve(index, chunks, metas, query, k=6)

        # 5. 컨텍스트 생성 및 LLM 호출 (Generate)
        context = llm.to_context(hits)
        
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"<컨텍스트>\n{context}\n\n[질문]\n{query}"
                f"\n\n형식:\n- 요약(2~5문장)\n- 핵심포인트(불릿)\n- 인용(제목/출처)\n- 면책고지")},
            {"role": "system", "content": "최종 답만 한국어로. 사고흐름/메타 금지."}
        ]
        
        answer = llm.call_llm(messages)

        # 6. 결과 및 인용 출력
        citations = [
            {
                "title": h[1]["meta"].get("title"),
                "source": h[1]["meta"].get("source"),
                "score": round((h[0] + 1) / 2, 3) # IP -> Cosine
            } for h in hits
        ]
        
        print("\n=== 답변 ===\n", answer.strip())
        print("\n=== 출처 및 정확도 ===")
        for c in citations:
            print(f"- {c['title']} ({c['source']}) | Score: {c['score']:.3f}")

if __name__ == "__main__":
    main()