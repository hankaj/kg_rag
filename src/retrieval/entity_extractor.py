from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.models.entities import Entities
from src.config.settings import DEFAULT_LLM_MODEL


class EntityExtractor:
    """Extracts entities from text using LLM"""
    
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the entity extractor
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )
        self.chain = self.prompt | self.llm.with_structured_output(Entities)
        
    def extract(self, text: str) -> Entities:
        """
        Extract entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Entities object containing extracted entity names
        """
        return self.chain.invoke({"question": text})