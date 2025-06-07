from langchain_neo4j import Neo4jGraph
from typing import List, Dict, Any
from src.config.settings import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    BASE_ENTITY_LABEL,
    INCLUDE_SOURCE,
)
from neo4j import GraphDatabase


class GraphDatabase:
    """Interface for Neo4j graph database operations"""

    def __init__(self):
        """Initialize the graph database connection"""
        self.graph = Neo4jGraph(
            url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD
        )

    def query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            Query results
        """
        return self.graph.query(query, params)

    def add_graph_documents(self, graph_documents: List[Dict[str, Any]]) -> None:
        """
        Add graph documents to the database

        Args:
            graph_documents: List of graph documents
        """
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=BASE_ENTITY_LABEL,
            include_source=INCLUDE_SOURCE,
        )

    def create_fulltext_index(
        self, index_name: str, label: str, properties: List[str]
    ) -> None:
        """
        Create a fulltext index

        Args:
            index_name: Name of the index
            label: Node label to index
            properties: Node properties to index
        """
        properties_str = ", ".join([f"n.{prop}" for prop in properties])
        self.query(
            f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON EACH [{properties_str}]"
        )

    def create_entity_index(self):
        """Create the entity fulltext index"""
        self.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )
