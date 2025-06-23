from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
from langchain.schema import Document
from src.config.settings import DEFAULT_LLM_MODEL, GROQ_API_KEY, GROQ_API_URL


class GraphTransformer:

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_URL,
        )
        self.transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=True)

    def transform(self, documents: List[Document]) -> List[Dict[str, Any]]:
        documents = self.transformer.convert_to_graph_documents(documents)
        for doc in documents:
            for node in doc.nodes:
                if node.type == "":
                    node.type = "Other"
        return documents
