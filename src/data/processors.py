from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from typing import List
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


class TextProcessor:

    def process(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError("Subclasses must implement this method")


class DocumentSplitter(TextProcessor):
    """Splits documents into smaller chunks"""

    def __init__(
        self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def process(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)
