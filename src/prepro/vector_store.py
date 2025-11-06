from langchain_huggingface import HuggingFaceEmbeddings

def create_or_load_vector_store(
    documents: list,
    embeddings: HuggingFaceEmbeddings,
    store_path: str = "data/vectorstore"
) -> "FAISS":
    if os.path.exists(store_path):
        print(f"loading FAISS from {store_path}")
        vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("creating new FAISS index...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(store_path)
        print(f"saved in {store_path}")
    return vectorstore
