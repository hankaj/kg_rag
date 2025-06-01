from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from typing import List, Tuple
from src.graph.database import GraphDatabase
from src.retrieval.entity_extractor import EntityExtractor
from src.retrieval.query_utils import generate_full_text_query
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
from src.config.settings import DEFAULT_EMBEDDING_MODEL
import numpy as np
from langchain_core.messages import HumanMessage


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
        return [doc.page_content[7:] for doc in docs]


class HybridRetriever:
    
    def __init__(self):
        self.structured_retriever = StructuredRetriever()
        self.unstructured_retriever = StandardRetriever()
        
    def retrieve(self, question: str) -> Tuple[str, List[str]]:
        structured_data = self.structured_retriever.retrieve(question)
        unstructured_data = self.unstructured_retriever.retrieve(question)
        
        final_data = f"""Structured data:
{structured_data}

Unstructured data:
{"#Document ".join(unstructured_data)}
        """
        return final_data
    
class KGRetriever:
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.entity_extractor = EntityExtractor()
        self.graph_db = GraphDatabase()
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def retrieve(self, question: str) -> str:
        entities = self.entity_extractor.extract(question)
        document_candidates = []
        seen_doc_ids = set()
        print("Entities:")
        print(entities.names)

        for entity in entities.names:
            # Expanded retrieval with related entities and documents
            cypher = """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:3}) YIELD node, score

            CALL (node) {
                MATCH (node)-[*1..2]-(related)
                WHERE related:Entity
                WITH collect(DISTINCT related) AS related_list
                WITH related_list + [node] AS all_entities
                UNWIND all_entities AS ent
                MATCH (ent)<-[:MENTIONS]-(doc:Document)
                RETURN DISTINCT doc.id AS id, doc.text AS text
            }

            RETURN id, text
            LIMIT 20

            """
            query = generate_full_text_query(entity)
            response = self.graph_db.query(cypher, {"query": query})
            for doc in response:
                if doc["id"] not in seen_doc_ids:
                    seen_doc_ids.add(doc["id"])
                    document_candidates.append(doc)

        if not document_candidates:
            return ["No relevant documents found."]
        
        # Embed question and documents
        question_embedding = np.array(self.embeddings.embed_query(question))
        doc_texts = [doc['text'] for doc in document_candidates]
        doc_embeddings = np.array(self.embeddings.embed_documents(doc_texts))  # shape: (N, dim)

        # Compute cosine similarity
        similarities = np.dot(doc_embeddings, question_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-10
        )
        top_indices = np.argsort(similarities)[-4:][::-1]  # Top 2 indices descending

        # Return top 2 documents
        top_docs = [doc_texts[i] for i in top_indices]

        return top_docs
    
import numpy as np
from typing import List, Dict

class SmartKGRetriever:
    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        self.entity_extractor = EntityExtractor()
        self.graph_db = GraphDatabase()
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
    def retrieve(self, question: str) -> List[str]:
        # Get more candidates from graph
        candidates = self._get_candidates(question)
        if not candidates:
            return []
        
        # Score and rank
        ranked_docs = self._rank_documents(question, candidates)
        
        # Smart selection instead of just top-2
        selected = self._select_best_docs(ranked_docs)
        
        return [doc['text'] for doc in selected]
    
    def _get_candidates(self, question: str) -> List[Dict]:
        """Get candidates from graph with better coverage"""
        entities = self.entity_extractor.extract(question)
        candidates = []
        
        for entity in entities.names:
            # Enhanced query to get more relevant docs
            cypher = """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit:5}) YIELD node, score
            MATCH (node)<-[:MENTIONS]-(doc:Document)
            
            OPTIONAL MATCH (node)-[:RELATED_TO*1..2]-(related:Entity)
            OPTIONAL MATCH (related)<-[:MENTIONS]-(related_doc:Document)
            
            WITH collect(DISTINCT doc) + collect(DISTINCT related_doc) AS all_docs
            UNWIND all_docs AS d
            
            RETURN DISTINCT d.id AS id, d.text AS text
            LIMIT 15
            """
            
            result = self.graph_db.query(cypher, {"query": generate_full_text_query(entity)})
            candidates.extend(result)
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for doc in candidates:
            if doc['id'] not in seen:
                seen.add(doc['id'])
                unique_candidates.append(doc)
        
        return unique_candidates[:20]  # Cap at 20 candidates
    
    def _rank_documents(self, question: str, candidates: List[Dict]) -> List[Dict]:
        """Rank documents by combined relevance score"""
        if not candidates:
            return []
        
        # Get embeddings
        question_embedding = np.array(self.embeddings.embed_query(question))
        doc_texts = [doc['text'] for doc in candidates]
        doc_embeddings = np.array(self.embeddings.embed_documents(doc_texts))
        
        # Calculate semantic similarity
        similarities = np.dot(doc_embeddings, question_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-10
        )
        
        # Add scores to documents and sort
        for i, doc in enumerate(candidates):
            doc['score'] = similarities[i]
        
        return sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    def _select_best_docs(self, ranked_docs: List[Dict]) -> List[Dict]:
        """Smart selection instead of just top-2"""
        if not ranked_docs:
            return []
        
        # Always include the best document
        selected = [ranked_docs[0]]
        
        # Determine how many more to include based on score gaps
        for i in range(1, min(len(ranked_docs), 6)):  # Max 6 docs
            current_score = ranked_docs[i]['score']
            top_score = ranked_docs[0]['score']
            
            # Include if score is decent (above threshold)
            if current_score > 0.3:
                # Check if it's too similar to already selected docs
                if not self._too_similar(ranked_docs[i], selected):
                    selected.append(ranked_docs[i])
            
            # Stop if score drops significantly
            if top_score - current_score > 0.4:
                break
            
            # Stop if we have enough good documents
            if len(selected) >= 4 and current_score < 0.5:
                break
        
        # Ensure we have at least 2 documents if available
        if len(selected) == 1 and len(ranked_docs) > 1:
            if not self._too_similar(ranked_docs[1], selected):
                selected.append(ranked_docs[1])
        
        return selected
    
    def _too_similar(self, doc: Dict, selected_docs: List[Dict]) -> bool:
        """Check if document is too similar to already selected ones"""
        # if not selected_docs:
        #     return False
        
        # doc_embedding = np.array(self.embeddings.embed_query(doc['text']))
        
        # for selected in selected_docs:
        #     selected_embedding = np.array(self.embeddings.embed_query(selected['text']))
        #     similarity = np.dot(doc_embedding, selected_embedding) / (
        #         np.linalg.norm(doc_embedding) * np.linalg.norm(selected_embedding) + 1e-10
        #     )
            
        #     # If too similar to any selected doc, skip it
        #     if similarity > 0.8:
        #         return True
        
        return False
    
class RerankRetriever:
    def __init__(self, llm: ChatOpenAI):
        self.structured_retriever = SmartKGRetriever()
        self.standard_retriever = StandardRetriever()
        self.llm = llm
        
    def retrieve(self, question: str) -> Tuple[str, List[str]]:
        standard_data = self.structured_retriever.retrieve(question)
        kg_data = self.standard_retriever.retrieve(question)
        combined_results = list(set(standard_data + kg_data))
        chosen_docs = self.select_relevant_documents(question, combined_results)
        return chosen_docs
    
    def select_relevant_documents(self, question: str, documents: List[str]) -> List[str]:
        doc_list_str = "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(documents)])
        prompt = (
            f"Question: {question}\n\n"
            f"Documents:\n{doc_list_str}\n\n"
            "Select the documents that are most helpful for answering the question. "
            "Respond with a list of the document numbers (e.g., [1, 3, 5]). Response should contain only list of numbers."
        )

        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            selected_indices = eval(response.content)
        except:
            selected_indices = []

        selected_docs = [documents[i - 1] for i in selected_indices if 0 < i <= len(documents)]
        return selected_docs