from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, Tuple, Optional
from src.retrieval.hybrid_retriever import HybridRetriever
from src.chains.chat_history import ChatHistoryManager
from src.config.settings import DEFAULT_LLM_MODEL


class QAChainHistory:
    """Question-answering chain with chat history support"""
    
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the QA chain
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.llm = ChatOpenAI(temperature=0, model_name=model_name)
        self.retriever = HybridRetriever()
        self.chat_history_manager = ChatHistoryManager(model_name=model_name)
        
        self._setup_chain()
        
    def _setup_chain(self):
        """Set up the QA chain components"""
        # Template for the final prompt
        template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
        qa_prompt = ChatPromptTemplate.from_template(template)
        
        # Search query preprocessing
        search_query = RunnableBranch(
            # If input includes chat_history, we condense it with the follow-up question
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))),
                RunnableLambda(
                    lambda x: self.chat_history_manager.condense_question(
                        x["question"], x["chat_history"]
                    )
                ),
            ),
            # Else, we have no chat history, so just pass through the question
            RunnableLambda(lambda x: x["question"]),
        )
        
        # Define retrieval function
        retrieval_function = lambda query: self.retriever.retrieve(query)
        
        # Construct the full chain
        self.chain = (
            RunnableParallel(
                {
                    "context": search_query | retrieval_function,
                    "question": RunnablePassthrough(),
                }
            )
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(self, input_dict: Dict[str, Any]) -> str:
        """
        Invoke the QA chain
        
        Args:
            input_dict: Input dictionary with 'question' and optional 'chat_history'
            
        Returns:
            Answer to the question
        """
        return self.chain.invoke(input_dict)
    
    def run(self, question: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """
        Run the QA chain with a question and optional chat history
        
        Args:
            question: User question
            chat_history: Optional list of (human, ai) message tuples
            
        Returns:
            Answer to the question
        """
        input_dict = {"question": question}
        if chat_history:
            input_dict["chat_history"] = chat_history
            
        return self.invoke(input_dict)
