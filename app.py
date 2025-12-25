import os
import streamlit as st
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain.chains import RetrievalQA
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Resume Search", layout="wide")
st.title("📄 Resume Search ChatBot")
st.write("Upload resumes (PDF) and search candidates by skills")

# -----------------------------
# Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Vector DB
# -----------------------------
PERSIST_DIR = "chroma_db"

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview",
                             google_api_key=os.getenv("GOOGLE_API_KEY"),
                             temperature=0.2)

uploaded_files = st.file_uploader("Upload Resume PDFs",type="pdf",accept_multiple_files=True)

if uploaded_files:
    documents = []

    for pdf in uploaded_files:
        with open(pdf.name, "wb") as f:
            f.write(pdf.getbuffer())

        loader = PyPDFLoader(pdf.name)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    # -----------------------------
    # Store in Chroma
    # -----------------------------
    vectordb = Chroma.from_documents(split_docs,embeddings,persist_directory=PERSIST_DIR)
    vectordb.persist()

    st.success("✅ Resumes indexed successfully!")

if os.path.exists(PERSIST_DIR):
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # qa_chain = retrieval_qa.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff")
    qa_chain = RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff")
 
    # -----------------------------
    # Query Input
    # -----------------------------
    query = st.text_input("🔍 Enter skill / role to search candidates")

    if query:
        response = qa_chain.run(query)
        st.subheader("📌 Matching Candidates")
        st.write(response)