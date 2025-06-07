from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document
from typing import List
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


class TextProcessor:
    """Base class for text processors"""

    def process(self, documents: List[Document]) -> List[Document]:
        """Process documents"""
        raise NotImplementedError("Subclasses must implement this method")


class DocumentSplitter(TextProcessor):
    """Splits documents into smaller chunks"""

    def __init__(
        self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the document splitter

        Args:
            chunk_size: The size of each chunk
            chunk_overlap: The overlap between chunks
        """
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def process(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        return self.text_splitter.split_documents(documents)
