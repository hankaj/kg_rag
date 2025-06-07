from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from src.config.settings import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
import os


class GraphVisualizer:
    """Visualizes Neo4j graphs"""

    def __init__(self):
        """Initialize the graph visualizer"""
        self.uri = NEO4J_URI
        self.username = NEO4J_USERNAME
        self.password = NEO4J_PASSWORD

    def create_widget(self, cypher: str) -> GraphWidget:
        """
        Create a graph widget for visualization

        Args:
            cypher: Cypher query to visualize

        Returns:
            Graph widget
        """
        driver = GraphDatabase.driver(uri=self.uri, auth=(self.username, self.password))
        session = driver.session()
        widget = GraphWidget(graph=session.run(cypher).graph())
        widget.node_label_mapping = "id"
        session.close()
        driver.close()
        return widget

    def show_graph(
        self, cypher: str = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"
    ) -> GraphWidget:
        """
        Show the graph based on a Cypher query

        Args:
            cypher: Cypher query to visualize

        Returns:
            Graph widget
        """
        return self.create_widget(cypher)
