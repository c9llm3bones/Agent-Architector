from typing import List
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from .base_loader import BaseLoader

class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> List[Document]:
        reader = PdfReader(file_path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": i}
                ))
        return docs