from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.models.entities import Entities
from src.config.settings import OPENAI_LLM_MODEL, GROQ_API_KEY, GROQ_API_URL


class EntityExtractor:
    
    def __init__(self, model_name: str = OPENAI_LLM_MODEL):
        self.llm = ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            temperature=0,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a state-of-the-art entity extractor used to identify key entities "
                    "from natural language questions for knowledge graph querying. "
                    "Your goal is to extract all relevant named entities, concepts, or keywords "
                    "that may exist in a knowledge graph, in a structured format. "
                    "Only return entities that are significant for understanding or answering the question. "
                    "Do not return generic terms, stopwords, or question words. "
                    "Output should strictly follow the given schema without explanation."
                ),
                (
                    "human",
                    "Extract all entities from the following question "
                    "input: {question}",
                ),
            ]
        )
        self.chain = self.prompt | self.llm.with_structured_output(Entities)
        
    def extract(self, text: str) -> Entities:
        return self.chain.invoke({"question": text})