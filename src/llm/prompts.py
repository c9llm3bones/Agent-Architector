from langchain_core.prompts import ChatPromptTemplate

def get_qa_prompt() -> ChatPromptTemplate:
    template = """
You are AI-architect, a specialist in technical documentation.

Context:
{context}

Question:
{input}
"""
    return ChatPromptTemplate.from_template(template)