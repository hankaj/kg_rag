from src.data.loaders import WikipediaDataLoader
from src.data.processors import DocumentSplitter
from src.graph.database import GraphDatabase
from src.graph.transformers import GraphTransformer
from src.graph.visualization import GraphVisualizer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector

def build_knowledge_graph():
    """Build a knowledge graph from Wikipedia data"""
    print("Loading data from Wikipedia...")
    data_loader = WikipediaDataLoader(query="Elizabeth I", load_max_docs=3)
    raw_documents = data_loader.load()
    
    print("Splitting documents...")
    document_splitter = DocumentSplitter()
    documents = document_splitter.process(raw_documents)
    
    print("Transforming documents to graph structure...")
    graph_transformer = GraphTransformer()
    graph_documents = graph_transformer.transform(documents)
    
    print("Adding documents to graph database...")
    graph_db = GraphDatabase()
    graph_db.add_graph_documents(graph_documents)
    
    print("Creating entity index...")
    graph_db.create_entity_index()
    
    print("Creating vector embeddings...")
    embeddings = OpenAIEmbeddings()
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    
    print("Graph building complete!")
    
    # Visualize the graph
    print("Generating graph visualization...")
    visualizer = GraphVisualizer()
    widget = visualizer.show_graph()
    print("Graph visualization widget created.")
    
    return widget

if __name__ == "__main__":
    build_knowledge_graph()
