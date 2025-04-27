from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from langchain.schema import Document
from src.config.settings import DEFAULT_LLM_MODEL


class GraphTransformer:
    """Transforms documents into graph structures"""
    
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the graph transformer
        
        Args:
            model_name: Name of the LLM model to use
        """
        llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.transformer = LLMGraphTransformer(llm=llm)
        
    def transform(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Transform documents into graph documents
        
        Args:
            documents: List of documents to transform
            
        Returns:
            List of graph documents
        """
        return self.transformer.convert_to_graph_documents(documents)