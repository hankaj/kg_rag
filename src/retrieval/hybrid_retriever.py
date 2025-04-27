from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from typing import List, Dict, Any
from src.graph.database import GraphDatabase
from src.retrieval.entity_extractor import EntityExtractor
from src.retrieval.query_utils import generate_full_text_query
from src.config.settings import DEFAULT_EMBEDDING_MODEL


class StructuredRetriever:
    """Retrieves information from the graph database using entity extraction"""
    
    def __init__(self):
        """Initialize the structured retriever"""
        self.entity_extractor = EntityExtractor()
        self.graph_db = GraphDatabase()
        
    def retrieve(self, question: str) -> str:
        """
        Retrieve structured information based on entities in the question
        
        Args:
            question: User question
            
        Returns:
            Retrieved structured information
        """
        result = ""
        entities = self.entity_extractor.extract(question)
        
        for entity in entities.names:
            response = self.graph_db.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL (node, node) {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response]) + "\n"
            
        return result.strip()


class UnstructuredRetriever:
    """Retrieves information using vector similarity search"""
    
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize the unstructured retriever
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = Neo4jVector.from_existing_graph(
            self.embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
    def retrieve(self, question: str, k: int = 4) -> List[str]:
        """
        Retrieve unstructured information based on vector similarity
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved document contents
        """
        docs = self.vector_store.similarity_search(question, k=k)
        return [doc.page_content for doc in docs]


class HybridRetriever:
    """Combines structured and unstructured retrieval methods"""
    
    def __init__(self):
        """Initialize the hybrid retriever"""
        self.structured_retriever = StructuredRetriever()
        self.unstructured_retriever = UnstructuredRetriever()
        
    def retrieve(self, question: str) -> str:
        """
        Retrieve information using both structured and unstructured methods
        
        Args:
            question: User question
            
        Returns:
            Combined retrieval results
        """
        structured_data = self.structured_retriever.retrieve(question)
        unstructured_data = self.unstructured_retriever.retrieve(question)
        
        final_data = f"""Structured data:
{structured_data}

Unstructured data:
{"#Document ".join(unstructured_data)}
        """
        
        return final_data