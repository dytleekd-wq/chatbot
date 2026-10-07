"""DATA 폴더의 문서를 검색해 답변하는 Streamlit RAG 챗봇입니다."""

from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# 프로젝트 최상위 폴더와 DATA 폴더를 기준으로 문서를 찾습니다.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "DATA"
# 코사인 유사도가 이 값보다 낮으면 문서와 관련 없는 질문으로 판단합니다.
# InMemoryVectorStore의 검색 점수는 1에 가까울수록 유사합니다.
MIN_RELEVANCE_SCORE = 0.48


def read_pdf_documents() -> list[Document]:
    """DATA 폴더의 모든 PDF를 페이지 단위 LangChain 문서로 읽습니다."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA 폴더를 찾을 수 없습니다: {DATA_DIR}")

    documents: list[Document] = []
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"DATA 폴더에 PDF 파일이 없습니다: {DATA_DIR}")

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number,
                        },
                    )
                )

    if not documents:
        raise ValueError("PDF에서 읽을 수 있는 텍스트를 찾지 못했습니다.")
    return documents


def format_documents(documents: list[Document]) -> str:
    """검색된 문서를 LLM이 읽을 수 있는 하나의 문맥으로 합칩니다."""
    return "\n\n--- 문서 구분 ---\n\n".join(
        f"출처: {document.metadata.get('source')} (p.{document.metadata.get('page')})\n"
        f"내용:\n{document.page_content}"
        for document in documents
    )


def format_chat_history(messages: list[dict[str, Any]], limit: int = 6) -> str:
    """최근 대화를 검색어 보정용 텍스트로 바꿉니다.

    대화 내용은 '그 경우', '그 금액' 같은 후속 질문의 대상을 파악하는 데만 쓰고,
    답변의 사실 근거는 항상 DATA 문서 검색 결과로 제한합니다.
    """
    if not messages:
        return "이전 대화 없음"

    role_name = {"user": "사용자", "assistant": "챗봇"}
    return "\n".join(
        f"{role_name.get(message['role'], message['role'])}: {message['content']}"
        for message in messages[-limit:]
    )


def evidence_sentence(text: str, limit: int = 500) -> str:
    """검색 문서에서 화면에 보여줄 근거 문장을 짧게 뽑습니다."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    evidence = " ".join(sentences[:2]) if sentences else text.strip()
    return evidence[:limit] + ("…" if len(evidence) > limit else "")


def configure_openai_api_key() -> bool:
    """로컬 .env 또는 Streamlit Cloud Secrets에서 API 키를 준비합니다."""
    # 로컬 개발 환경에서는 프로젝트 최상위의 .env 파일을 사용합니다.
    load_dotenv(PROJECT_ROOT / ".env")
    if os.getenv("OPENAI_API_KEY"):
        return True

    # Streamlit Cloud에서는 Settings > Secrets에 저장한 값을 사용합니다.
    # secrets.toml 파일이 없는 로컬 환경에서도 오류가 나지 않도록 처리합니다.
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        api_key = None

    if api_key:
        os.environ["OPENAI_API_KEY"] = str(api_key)
        return True
    return False


@st.cache_resource(show_spinner="문서를 읽고 벡터 DB를 준비하는 중입니다...")
def build_rag_components() -> tuple[Any, Any, Any, int]:
    """문서 분할, 임베딩, InMemoryVectorStore, 답변 체인을 준비합니다."""
    # .env의 OPENAI_API_KEY는 langchain-openai가 자동으로 읽습니다.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    pages = read_pdf_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    vector_store = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
    # 최신 Runnable(LCEL)로 이전 대화를 독립적인 문서 검색 질문으로 보정합니다.
    search_query_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 문서 검색어 보정 도우미입니다.
이전 대화는 대명사와 생략된 조건을 해석하는 용도로만 사용하세요.
현재 질문을 문서 검색에 적합한 하나의 독립적인 한국어 질문으로 바꾸세요.
현재 질문이 독립적이면 그대로 반환하세요.
답변하거나 새로운 사실을 추가하지 말고, 검색 질문만 출력하세요.

이전 대화:
{chat_history}""",
            ),
            ("human", "현재 질문: {question}"),
        ]
    )
    search_query_chain = search_query_prompt | llm | StrOutputParser()

    # 구버전 RetrievalQA/LLMChain 대신 최신 Runnable(LCEL) 방식으로 답변을 생성합니다.
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 공무원 여비 문서 전문 답변 도우미입니다.
반드시 아래 문서 내용만 근거로 답변하세요.
이전 대화는 후속 질문의 대상을 해석하는 용도일 뿐, 사실의 근거로 사용하면 안 됩니다.
검색된 문서가 질문과 무관하면 '문서에서 확인할 수 없습니다.'라고 답하세요.
질문과 관련된 문서는 있지만 정확한 금액·적용 여부를 판단할 정보가 부족한 경우에는
'정확한 산정을 위해 추가 정보가 필요합니다.'라고 먼저 안내하세요.
그 다음 문서에서 확인되는 원칙을 설명하고, 필요한 정보(예: 직급, 교통수단과 실제 운임,
숙박 지역과 실제 숙박비, 출장 일수)를 구체적으로 질문하세요.
추측, 일반 상식 보충, 문서에 없는 숫자나 규정 생성은 금지합니다.
답변은 한국어로 간결하고 명확하게 작성하세요.

이전 대화:
{chat_history}

문서 내용:
{context}""",
            ),
            ("human", "질문: {question}"),
        ]
    )
    answer_chain = answer_prompt | llm | StrOutputParser()
    return search_query_chain, answer_chain, vector_store, len(chunks)


def main() -> None:
    """Streamlit 화면과 질의 처리 흐름을 실행합니다."""
    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚", layout="centered")
    st.title("📚 공무원 여비 RAG 챗봇")
    st.caption("DATA 폴더의 문서만 검색하여 답변합니다.")

    if not configure_openai_api_key():
        st.error("OPENAI_API_KEY가 없습니다. 로컬 .env 또는 Streamlit Cloud Secrets에 API 키를 설정해 주세요.")
        st.stop()

    try:
        search_query_chain, answer_chain, vector_store, chunk_count = build_rag_components()
    except Exception as error:  # 사용자에게 실행 원인을 알기 쉽게 보여줍니다.
        st.error(f"문서 또는 OpenAI 설정을 준비하지 못했습니다: {error}")
        st.stop()

    with st.sidebar:
        st.header("설정")
        st.write(f"검색 문서 조각: {chunk_count}개")
        if st.button("문서 다시 읽기"):
            build_rag_components.clear()
            st.rerun()
        if st.button("대화 초기화"):
            st.session_state.messages = []
            st.rerun()

    # Streamlit은 상호작용할 때마다 화면을 다시 그리므로, 대화 내용을 세션에 보관합니다.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 새로고침 전까지 저장된 대화 내용을 화면에 다시 표시합니다.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                st.markdown("**출처 및 근거**")
                for source in message["sources"]:
                    st.markdown(
                        f"- `{source['file']}` p.{source['page']}: {source['evidence']}"
                    )

    question = st.chat_input("문서에 대해 궁금한 내용을 질문하세요.")
    if not question:
        if not st.session_state.messages:
            st.info("예: 출장 여비 지급 기준은 어떻게 되나요?")
        return

    # 새 질문을 저장하기 전의 대화만 사용해야 현재 질문이 중복되지 않습니다.
    chat_history = format_chat_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("문서에서 근거를 찾는 중입니다..."):
            # 후속 질문을 독립 검색어로 보정한 후, 그 검색어로만 문서를 찾습니다.
            search_query = search_query_chain.invoke(
                {"chat_history": chat_history, "question": question}
            )
            # 항상 결과를 반환하는 retriever 대신, 유사도 점수를 확인해 관련 문서만 사용합니다.
            scored_documents = vector_store.similarity_search_with_score(search_query, k=8)
            relevant_documents = [
                document
                for document, score in scored_documents
                if score >= MIN_RELEVANCE_SCORE
            ]

            if relevant_documents:
                context = format_documents(relevant_documents)
                answer = answer_chain.invoke(
                    {
                        "chat_history": chat_history,
                        "question": question,
                        "context": context,
                    }
                )
            else:
                # 관련 문서가 없으면 LLM을 호출하지 않아 추측 답변도 막습니다.
                answer = "문서에서 확인할 수 없습니다."
                relevant_documents = []
        st.markdown(answer)

        seen: set[tuple[str, int]] = set()
        sources: list[dict[str, Any]] = []
        if relevant_documents:
            st.markdown("**출처 및 근거**")
            for document in relevant_documents:
                source = str(document.metadata.get("source", "알 수 없음"))
                page = int(document.metadata.get("page", 0))
                key = (source, page)
                if key in seen:
                    continue
                seen.add(key)
                evidence = evidence_sentence(document.page_content)
                sources.append({"file": source, "page": page, "evidence": evidence})
                st.markdown(f"- `{source}` p.{page}: {evidence}")

        # 답변과 출처를 함께 저장해 다음 화면 갱신에도 대화를 유지합니다.
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
