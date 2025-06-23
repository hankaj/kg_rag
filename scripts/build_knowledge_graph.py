import argparse
import time
from src.data.loaders import WikipediaDataLoader, HotpotQADataLoader
from src.data.processors import DocumentSplitter
from src.graph.database import GraphDatabase
from src.graph.transformers import GraphTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector


def build_knowledge_graph(
    source: str,
    query: str = None,
    split: str = "train",
    max_docs: int = None,
    batch_size: int = 50,
    start_from: int = 0,
):
    """
    Build a knowledge graph from either Wikipedia or HotpotQA with batch processing
    and checkpointing for resumable execution

    Args:
        source: Data source ('wikipedia' or 'hotpotqa')
        query: Wikipedia search query (only for Wikipedia source)
        split: HotpotQA split to use (train/dev/test)
        max_docs: Maximum number of documents to load
        batch_size: Number of documents to process in each batch
        start_from: Index of document to start processing from
    """
    print(f"Loading data from source='{source}'...")
    if source == "wikipedia":
        if not query:
            raise ValueError("Wikipedia source requires --query argument.")
        data_loader = WikipediaDataLoader(query=query, load_max_docs=max_docs)
    elif source == "hotpotqa":
        data_loader = HotpotQADataLoader(split=split, load_max_docs=max_docs)
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'wikipedia' or 'hotpotqa'.")

    raw_documents = data_loader.load()
    total_docs = len(raw_documents)
    print(f"Loaded {total_docs} raw documents")

    # Skip documents if starting from a specific index
    if start_from > 0:
        if start_from >= total_docs:
            raise ValueError(
                f"start_from={start_from} exceeds the number of documents ({total_docs})"
            )
        print(f"Skipping first {start_from} documents...")
        raw_documents = raw_documents[start_from:]
        total_docs = len(raw_documents)

    # Calculate number of batches
    num_batches = (total_docs + batch_size - 1) // batch_size  # Ceiling division
    print(
        f"Processing {total_docs} documents in {num_batches} batches (batch_size={batch_size})..."
    )

    # Initialize components
    document_splitter = DocumentSplitter()
    graph_transformer = GraphTransformer()
    graph_db = GraphDatabase()

    # Process in batches
    start_doc_idx = start_from
    try:
        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, total_docs)
            batch_docs = raw_documents[batch_start:batch_end]

            print(
                f"\nProcessing batch {batch_num + 1}/{num_batches} (documents {start_doc_idx + batch_start} to {start_doc_idx + batch_end - 1})..."
            )

            # Split documents
            print("Splitting documents...")
            documents = document_splitter.process(batch_docs)

            # Transform to graph structure
            print("Transforming documents to graph structure...")
            graph_documents = graph_transformer.transform(documents)

            # Add to graph database
            print("Adding documents to graph database...")
            graph_db.add_graph_documents(graph_documents)

            # Update checkpoint after each successful batch
            current_doc_idx = start_doc_idx + batch_end - 1
            print(f"Completed through document {current_doc_idx}.")

            # Add delay between batches to prevent overloading
            if batch_num < num_batches - 1:
                print(
                    f"Batch {batch_num + 1} complete. Pausing briefly before next batch..."
                )
                time.sleep(1)

    except Exception as e:
        print(
            f"\nError encountered during batch {batch_num + 1}: {e}. You need to start from {start_doc_idx + batch_start}"
        )
        print(
            "Stopping further processing and creating vector index with processed data so far..."
        )

    finally:
        print("\nCreating entity index...")
        graph_db.create_entity_index()

        print("Creating vector embeddings...")
        embeddings = OpenAIEmbeddings()
        Neo4jVector.from_existing_graph(
            embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )

        print("\nGraph building complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from Wikipedia or HotpotQA"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["wikipedia", "hotpotqa"],
        required=True,
        help="Data source: 'wikipedia' or 'hotpotqa'",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Wikipedia query (only for Wikipedia source)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HotpotQA split to use (train/dev/test)",
    )
    parser.add_argument(
        "--max_docs", type=int, default=None, help="Maximum number of documents to load"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of documents to process in each batch",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Index of document to start processing from (0-indexed)",
    )

    args = parser.parse_args()

    try:
        build_knowledge_graph(
            source=args.source,
            query=args.query,
            split=args.split,
            max_docs=args.max_docs,
            batch_size=args.batch_size,
            start_from=args.start_from,
        )
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Progress has been saved to checkpoint.")
        print("Run the script again with the same parameters to resume.")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        print("Progress has been saved to checkpoint.")
        print("Run the script again with the same parameters to resume.")
        # Re-raise if you want to see the full stack trace
        raise
