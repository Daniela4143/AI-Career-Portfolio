import streamlit as st
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# 加入短期記憶
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

# --- 1. 初始化與配置 ---
load_dotenv()

# 設定預設文件路徑 (與 V1 一致)
DOCUMENTS_PATH = "data/knowledge_sample.pdf"
VECTOR_DB_DIR = "chroma_db"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 檢查 API Key
if "GEMINI_API_KEY" not in os.environ:
    st.error("🚨 找不到 GEMINI_API_KEY！請在 .env 檔案中設定。")
    st.stop()

# --- 2. 緩存資源 (避免重複載入模型) ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)

@st.cache_resource
def setup_vectorstore():
    embeddings = get_embeddings()
    # 如果已有資料庫則加載，否則建立
    if os.path.exists(VECTOR_DB_DIR):
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    
    if os.path.exists(DOCUMENTS_PATH):
        loader = PyPDFLoader(DOCUMENTS_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        return Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=VECTOR_DB_DIR)
    
    return None

# --- 3. 建立對話鏈 (含記憶功能) ---
def create_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 初始化短期記憶
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

# --- 4. Streamlit 介面 ---
def main():
    st.set_page_config(page_title="AI 文件助手", layout="centered")
    st.title("🤖 企業級 RAG 問答機器人")
    st.info("本機器人具備「對話記憶」功能，您可以針對先前的回答繼續追問。")

    # 初始化知識庫
    vectorstore = setup_vectorstore()
    if not vectorstore:
        st.warning(f"請確保 `{DOCUMENTS_PATH}` 檔案存在。")
        return

    # 初始化對話鏈
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = create_chain(vectorstore)
    
    # 初始化聊天紀錄
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "你好！我是你的文件助手，請問今天想了解什麼？"}]

    # 顯示對話歷史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 使用者輸入
    if prompt := st.chat_input("請輸入您的問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = st.session_state.qa_chain.invoke({"question": prompt})
                answer = result["answer"]
                
                # 整理來源資訊
                sources = result.get("source_documents", [])
                source_text = "\n\n**📚 參考來源：**\n" + "\n".join([f"- 第 {doc.metadata.get('page','?')} 頁" for doc in sources])
                
                full_response = f"{answer}{source_text}"
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()