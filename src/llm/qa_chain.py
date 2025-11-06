import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from .prompts import get_qa_prompt

load_dotenv()

def create_rag_chain(vectorstore: FAISS, model_name: str = "mistral-small-latest"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    api_key = os.getenv("MISTRAL_API_KEY")
    llm = ChatMistralAI(
        model=model_name,
        mistral_api_key=api_key,
    )
    prompt = get_qa_prompt()
    #print(f"found context: {retriever}")
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain