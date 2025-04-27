from typing import List, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import DEFAULT_LLM_MODEL


class ChatHistoryManager:
    """Manages chat history and condenses follow-up questions"""
    
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the chat history manager
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
        self.condense_prompt = PromptTemplate.from_template(self.condense_template)
        
    def format_chat_history(self, chat_history: List[Tuple[str, str]]) -> List:
        """
        Format chat history into a list of messages
        
        Args:
            chat_history: List of (human, ai) message tuples
        
        Returns:
            List of formatted messages
        """
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer
    
    def condense_question(self, question: str, chat_history: List[Tuple[str, str]]) -> str:
        """
        Condense a follow-up question with chat history into a standalone question
        
        Args:
            question: Follow-up question
            chat_history: List of (human, ai) message tuples
            
        Returns:
            Standalone question
        """
        formatted_history = self.format_chat_history(chat_history)
        chain = self.condense_prompt | self.llm | StrOutputParser()
        return chain.invoke({"chat_history": formatted_history, "question": question})