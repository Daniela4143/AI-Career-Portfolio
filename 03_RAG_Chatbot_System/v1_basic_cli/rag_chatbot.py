
import os
from dotenv import load_dotenv # for loading .env configuration
from langchain_community.document_loaders import PyPDFLoader # for PDF document loading
from langchain_text_splitters import RecursiveCharacterTextSplitter # for text splitting
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # for Google GenAI models
from langchain_chroma import Chroma # for vector database
from langchain_core.prompts import PromptTemplate # for prompt template
from langchain_classic.chains.retrieval_qa.base import RetrievalQA # for retrieval QA chain
from langchain_huggingface import HuggingFaceEmbeddings # replace genai

# --- Configuration Section ---
# --- 初始化配置 ---
load_dotenv() # 自動加載同資料夾下的 .env

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("找不到 GEMINI_API_KEY，請確認 .env 檔案設定正確！")
# document path: supposed to be 'knowledge.pdf'
DOCUMENTS_PATH = "data/knowledge_sample.pdf"
VECTOR_DB_DIR = "chroma_db" # vector database save path

# other setting
RETRIEVER_TOP_K = 3 # Number of context chunks to retrieve (adjustable)
# -----------------------------------------------

def setup_rag_pipeline(doc_path: str, db_dir: str):
    """
    Set RAG Pipeline: Load documents, split into chunks, embed, and store in vector DB.
    """
    print("1. Starting building knowledge base...")

    # 1.1 Document Loading
    try:
        loader = PyPDFLoader(doc_path)
        documents = loader.load()
        print(f"Sucessfully loaded {len(documents)} pages from {doc_path}")
    except Exception as e:
        print(f"Error Loading document, please check the file path or document format: {e}")
        return None
    
    # 1.2 Text Splitting
    # use recursive splitter which is better for structured documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # max token length per chunk
        chunk_overlap=200   # overlap length between chunks, to maintain context
    )
    texts = text_splitter.split_documents(documents)
    print(f"Document split into {len(texts)} context chunks.")

    # 1.3 Embedding Model: transform text chunks into vector representations
    # use Google Embedding model *配額限制 (429 錯誤), 改用 Hugging Face
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Using local HuggingFace embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 1.4 Store in Vector DB
    # use ChromaDB to store text chunks and their vector embeddings
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=db_dir
    )
    print(f"Vector DB has been built and saved to {db_dir}.")

    return vectorstore

def create_retrieval_chain(vectorstore: Chroma):
    """
    Build Retrieval Chain and QA Prompt.
    """
    print("2. Building Retrieval QA Chain...")

    # 2.1 Initialize LLM model
    # use gemini-2.5-flash as LLM QA model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    # 2.2 Setup Retriever
    # acquire the TOP_K most revelant context chunks from vector DB
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    # 2.3 Setup Prompt Template (core of Prompt Engineering)
    # key to guide LLM to answer questions and use the preceding and following context that is retrieved
    # add some humanized and professional instructions to make the answer more friendly and reliable
    template = """
    Your are a professional AI application engineer. Your task is to answer user questions based on the provided context.
    Please guarantee that your answers are accurate, concise, and only based on the preceding and following context.
    If the context does not have enough information to answer the question, please respond with 'Sorry, I can't find the relevant information from the provided documents'
    
    Surrounding Context: {context}
    
    User Question: {question}
    
    Helpful and accurate Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # 2.4 Build Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # put all retrieved contect chunks into the prompt
        retriever=retriever,
        return_source_documents=True, # let user know the source documents of the answer
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain
    
def chat_loop(qa_chain: RetrievalQA):
    """
    loop to interact with RAG chatbot
    """
    print("3. Entering QA phase...")
    print("Enter 'exit' or 'quit' to end the chat.")

    while True:
        question = input("\nYour Question: ")
        if question.lower() in ['exit', 'quit']:
            print("Goodbye! RAG Chatbot has closed.")
            break
        if not question.strip():
            continue

        # run the Retrieval QA chain
        result = qa_chain.invoke({"query": question})

        # output the answer
        print("\n🤖 AI 回答:")
        print(result["result"])

        # Output source documents, which is important to debug and show RAG performance
        print("\n📚 資訊來源:")
        for doc in result["source_documents"]:
            print(f"- source document: {doc.metadata.get('source', 'N/A')}, page number: {doc.metadata.get('page', 'N/A')}")

def main():
    
    # 統一使用同一個 Embedding 設定
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(VECTOR_DB_DIR):
        print(f"1. 發現現有的知識庫，正在加載...")
        vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    else:
        vectorstore = setup_rag_pipeline(DOCUMENTS_PATH, VECTOR_DB_DIR)

    if vectorstore:
        qa_chain = create_retrieval_chain(vectorstore)
        chat_loop(qa_chain)

if __name__ == "__main__":
    main()