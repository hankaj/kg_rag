from langchain.document_loaders import WikipediaLoader, DirectoryLoader
from typing import List, Optional
from langchain.schema import Document
import os
from datasets import load_dataset

class DataLoader:
    """Base class for data loaders"""
    
    def load(self) -> List[Document]:
        """Load data from source"""
        raise NotImplementedError("Subclasses must implement this method")


class WikipediaDataLoader(DataLoader):
    """Loads data from Wikipedia articles"""
    
    def __init__(self, query: str, load_max_docs: int = None):
        """
        Initialize the Wikipedia loader
        
        Args:
            query: The Wikipedia search query
            load_max_docs: Maximum number of documents to load
        """
        self.query = query
        self.load_max_docs = load_max_docs
        
    def load(self) -> List[Document]:
        """
        Load documents from Wikipedia
        
        Returns:
            List of documents
        """
        raw_documents = WikipediaLoader(query=self.query).load()
        if self.load_max_docs is not None:
            return raw_documents[:self.load_max_docs]
        return raw_documents


class FolderDataLoader(DataLoader):
    """Loads documents from a folder"""
    def __init__(self, folder_path: str, glob_pattern: str = "**/*.*", load_max_docs: Optional[int] = None):
        """
        Initialize the folder loader
        Args:
            folder_path: Path to the folder containing documents
            glob_pattern: Pattern to match files (default: all files in all subdirectories)
            load_max_docs: Maximum number of documents to load
        """
        self.folder_path = folder_path
        self.glob_pattern = glob_pattern
        self.load_max_docs = load_max_docs
        
    def load(self) -> List[Document]:
        """
        Load documents from folder
        Returns:
            List of documents
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder path does not exist: {self.folder_path}")
            
        loader = DirectoryLoader(
            path=self.folder_path,
            glob=self.glob_pattern
        )
        
        raw_documents = loader.load()
        
        if self.load_max_docs is not None:
            return raw_documents[:self.load_max_docs]
        return raw_documents



class HotpotQADataLoader:
    """Loads data from the HotpotQA dataset"""

    def __init__(self, load_max_docs: int = None, split: str = 'train'):
        """
        Initialize the HotpotQA loader

        Args:
            load_max_docs: Maximum number of documents to load
            split: The split of the dataset to load (train, dev, test)
        """
        self.load_max_docs = load_max_docs
        self.split = split

    def load(self) -> List[Document]:
        """
        Load documents from HotpotQA

        Returns:
            List of documents
        """
        dataset = load_dataset("hotpot_qa", "distractor", split=self.split)
        
        documents = []
        for item in dataset:
            combined_sentences = " ".join([" ".join(sentence_list) for sentence_list in item['context']['sentences']])
            
            documents.append(
                Document(
                    page_content=combined_sentences,
                    metadata={'question': item['question'], 'answer': item['answer']['text']}
                )
            )
        
        if self.load_max_docs is not None:
            return documents[:self.load_max_docs]
        
        return documents