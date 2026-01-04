
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# 加入短期記憶
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# --- 1. 配置區 ---
load_dotenv()
DOCUMENTS_PATH = "data/knowledge_sample.pdf"
VECTOR_DB_DIR = "chroma_db"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_TOP_K = 5

# --- 2. 緩存組件 ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)

@st.cache_resource
def setup_knowledge_base():
    embeddings = get_embeddings()
    # 如果資料庫已存在則加載
    if os.path.exists(VECTOR_DB_DIR):
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    
    # 不存在則重新建立
    if not os.path.exists(DOCUMENTS_PATH):
        return None

    loader = PyPDFLoader(DOCUMENTS_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    return Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=VECTOR_DB_DIR)

def create_qa_chain(vectorstore):
    if "GEMINI_API_KEY" not in os.environ:
        return None
    
    # 初始化 LLM 和檢索器
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    # 初始化短期記憶：這是 V3 的核心亮點
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False 
    )

# --- 3. Streamlit 介面邏輯 ---
def clear_history():
    if "messages" in st.session_state: del st.session_state.messages
    if "qa_chain" in st.session_state: del st.session_state.qa_chain

def main():
    st.set_page_config(page_title="Gemini RAG 記憶助手", layout="wide")
    st.title("🚀 Project: 高級文件問答機器人 (V3)")
    st.caption("具備對話記憶功能，能理解上下文連結。")

    with st.sidebar:
        st.button("🧹 清除對話紀錄", on_click=clear_history)
        if "GEMINI_API_KEY" not in os.environ:
            st.error("🚨 找不到 API Key，請檢查 .env")

    # 初始化知識庫
    vectorstore = setup_knowledge_base()
    if not vectorstore:
        st.warning(f"請上傳文件至 `{DOCUMENTS_PATH}`")
        st.stop()

    # 初始化對話鏈
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = create_qa_chain(vectorstore)
    
    qa_chain = st.session_state.qa_chain
    if not qa_chain:
        st.error("無法啟動問答鏈，請檢查 API Key。")
        st.stop()

    # 聊天訊息處理
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "您好！我已準備好回答關於文件的細節，我也會記得我們之前的對話內容。"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("請輸入您的問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考並檢索中..."):
                try:
                    # 調用對話鏈
                    result = qa_chain.invoke({"question": prompt})
                    answer = result["answer"]
                    sources = result.get("source_documents", [])
                    
                    # 格式化輸出來源
                    source_info = "\n\n---\n**📚 資訊來源：**\n" + "\n".join(
                        [f"- 第 {doc.metadata.get('page','?')} 頁" for doc in sources]
                    )
                    
                    full_response = f"{answer}{source_info}"
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"執行出錯: {e}")

if __name__ == "__main__":
    main()