from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from langchain.schema import Document
from src.config.settings import DEFAULT_LLM_MODEL, GROQ_API_KEY, GROQ_API_URL


class GraphTransformer:
    """Transforms documents into graph structures"""
    
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the graph transformer
        
        Args:
            model_name: Name of the LLM model to use
        """
        llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=0,
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_URL,
        )
        self.transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=True)
        
    def transform(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Transform documents into graph documents
        
        Args:
            documents: List of documents to transform
            
        Returns:
            List of graph documents
        """
        documents = self.transformer.convert_to_graph_documents(documents)
        for doc in documents:
            for node in doc.nodes:
                if node.type == '':
                    node.type = 'Other'
        return documents