from pathlib import Path
from typing import List
from langchain_core.documents import Document
from src.loaders.pdf_loader import PDFLoader
from src.loaders.markdown_loader import MarkdownLoader

def load_all_documents(data_dir: str = "data") -> List[Document]:
    docs = []
    pdf_loader = PDFLoader()
    md_loader = MarkdownLoader()

    # PDF
    pdf_dir = Path(data_dir) / "pdf"
    if pdf_dir.exists():
        for file in pdf_dir.glob("*.pdf"):
            docs.extend(pdf_loader.load(str(file)))
    # Markdown
    md_dir = Path(data_dir) / "markdown"
    if md_dir.exists():
        for file in md_dir.glob("*.md"):
            docs.extend(md_loader.load(str(file)))

    print(f"{len(docs)} pages of documents")
    return docs