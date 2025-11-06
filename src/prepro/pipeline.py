from src.utils.document_loader import load_all_documents
from src.prepro.split_text import split_documents
from src.prepro.embeddings import get_embeddings_model
from src.prepro.vector_store import create_or_load_vector_store

def build_vectorstore(data_dir: str = "data") -> None:
 
    docs = load_all_documents(data_dir)
    if not docs:
        raise ValueError("No documents for indexing!")
    
    chunks = split_documents(docs)
    print(f"splitted into {len(chunks)} chunks")

    embeddings = get_embeddings_model()

    vectorstore = create_or_load_vector_store(chunks, embeddings)
    return vectorstore