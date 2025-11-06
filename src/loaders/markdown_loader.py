from typing import List
from langchain_core.documents import Document
from .base_loader import BaseLoader

class MarkdownLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(
            page_content=text,
            metadata={"source": file_path}
        )]