import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from src import config, data_loader, vector_store, llm

def format_docs(docs):
    """검색된 문서를 프롬프트에 삽입할 문자열로 포맷합니다."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_rag_chain_with_source():
    """
    답변과 출처를 함께 반환하는 RAG 체인을 초기화합니다.
    Streamlit의 캐시 기능을 사용하여 리소스를 한 번만 로드합니다.
    """
    # 1. 데이터 로드 및 분할
    docs = data_loader.load_all_documents()
    if not docs:
        return None
    splits = data_loader.split_documents(docs)

    # 2. 임베딩 및 벡터 저장소 생성
    embeddings = vector_store.get_embedding_model()
    db = vector_store.build_vector_store(splits, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 6})

    # 3. LLM 인스턴스화
    llm_instance = llm.get_llm()

    # 4. 프롬프트 템플릿 정의
    template = """
"당신은 한국 법령 리서처입니다. 아래 <컨텍스트>만을 근거로 "
"간결하고 정확히 답변하고, 인용(제목/출처)을 제시하세요. "
"모르면 모른다고 말하세요."

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

    # 5. LCEL을 사용한 RAG 체인 구성 (출처 포함)
    
    # 답변 생성 부분
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm_instance
        | StrOutputParser()
    )

    # 출처(context)와 질문(question)을 받고, 답변(answer)을 생성하여 함께 반환
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source

# --- Streamlit UI 구성 ---

st.title("법률 RAG Q&A 시스템")
st.markdown("---")

# RAG 체인 로드
rag_chain = get_rag_chain_with_source()

if rag_chain is None:
    st.error(f"⚠️  데이터 파일을 찾을 수 없습니다. '{config.DATA_DIR}' 디렉토리에 .txt 또는 .pdf 파일을 넣어주세요.")
else:
    # 사용자 질문 입력
    query = st.text_input(
        "궁금한 법률 정보를 질문하세요:",
        placeholder="예: 근로기준법상 연차 유급휴가에 대해 알려줘"
    )

    if st.button("질문하기"):
        if query:
            with st.spinner("답변을 생성하는 중입니다..."):
                try:
                    # 체인 실행 (결과에 'answer'와 'context' 포함)
                    result = rag_chain.invoke(query)

                    st.markdown("#### 답변")
                    st.markdown(result["answer"].strip())
                    
                    # 답변 근거 (출처) 표시
                    st.markdown("---")
                    with st.expander("📂 답변 근거 보기"):
                        for doc in result["context"]:
                            st.markdown(f"**[출처: {doc.metadata.get('source', 'N/A')}]**")
                            st.markdown(doc.page_content)
                            st.markdown("---")

                except Exception as e:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
        else:
            st.warning("질문을 입력해주세요.")
