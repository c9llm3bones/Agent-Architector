from langchain_core.prompts import ChatPromptTemplate

def get_qa_prompt() -> ChatPromptTemplate:
    template = """
Ты — AI-архитектор, специалист по технической документации.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Если ответ не содержится в контексте — скажи: "Я не знаю."

Контекст:
{context}

Вопрос:
{input}
"""
    return ChatPromptTemplate.from_template(template)