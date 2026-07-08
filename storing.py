from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(chunks):
    global embeddings

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vector_store

def retrieve_vector_data():
    global embeddings
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings)

    return db