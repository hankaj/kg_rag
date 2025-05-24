from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from typing import List, Tuple
from src.graph.database import GraphDatabase
from src.retrieval.entity_extractor import EntityExtractor
from src.retrieval.query_utils import generate_full_text_query
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from src.config.settings import DEFAULT_EMBEDDING_MODEL


class StructuredRetriever:
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.graph_db = GraphDatabase()
        
    def retrieve(self, question: str) -> str:
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


class StandardRetriever:
    
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = Neo4jVector.from_existing_graph(
            self.embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        
    def retrieve(self, question: str, k: int = 4) -> Tuple[List[str], List[str]]:
        sanitized_question = remove_lucene_chars(question)
        docs = self.vector_store.similarity_search(sanitized_question, k=k)
        return [doc.page_content for doc in docs], [doc.metadata['question'] for doc in docs]


class HybridRetriever:
    
    def __init__(self):
        self.structured_retriever = StructuredRetriever()
        self.unstructured_retriever = StandardRetriever()
        
    def retrieve(self, question: str) -> Tuple[str, List[str]]:
        structured_data = self.structured_retriever.retrieve(question)
        unstructured_data, questions = self.unstructured_retriever.retrieve(question)
        
        final_data = f"""Structured data:
{structured_data}

Unstructured data:
{"#Document ".join(unstructured_data)}
        """
        return final_data, questions