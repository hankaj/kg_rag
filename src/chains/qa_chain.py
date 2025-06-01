from typing import Dict, Any, TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.retriever import HybridRetriever, StandardRetriever, KGRetriever, SmartKGRetriever, RerankRetriever
from src.config.settings import DEFAULT_LLM_MODEL, GROQ_API_KEY, GROQ_API_URL

class QAState(TypedDict):
    question: str
    context: Optional[str]
    answer: Optional[str]
    retrieved_questions: Optional[List[str]]


class QAChain:
    
    def __init__(self, retrieval: str = "standard", model_name: str = DEFAULT_LLM_MODEL):
        self.llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=0,
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_URL
        )
        if retrieval=="hybrid":
            self.retriever = HybridRetriever()
        if retrieval=="kg":
            self.retriever = SmartKGRetriever()
        if retrieval=="rerank":
            self.retriever = RerankRetriever(self.llm)
        else:
            self.retriever = StandardRetriever()
        
        qa_template = """Answer the question based only on the following context:
{context}

Question: {question}

Be as simple and concise as possible — prefer short direct answers over full explanations.
Answer:"""
        self.qa_prompt = ChatPromptTemplate.from_template(qa_template)
        
        self._setup_graph()
    
    def _setup_graph(self):
        
        def retrieve_context(state: QAState) -> Dict:
            context = self.retriever.retrieve(state["question"])
            return {"context": context}
        
        def generate_answer(state: QAState) -> Dict:
            prompt = self.qa_prompt.invoke({
                "context": state["context"],
                "question": state["question"]
            })
            
            answer = self.llm.invoke(prompt).content
            return {"answer": answer}
        
        graph = StateGraph(QAState)
        
        graph.add_node("retrieval_node", retrieve_context)
        graph.add_node("generation_node", generate_answer)
        
        graph.add_edge("retrieval_node", "generation_node")
        graph.add_edge("generation_node", END)
        
        graph.set_entry_point("retrieval_node")
        
        self.graph = graph.compile()
    
    def invoke(self, input_dict: Dict[str, Any]) -> str:
        state = QAState(
            question=input_dict["question"],
            context=None,
            answer=None
        )
        
        result = self.graph.invoke(state)
        
        return result
    
    def run(self, question: str) -> str:
        return self.invoke({"question": question})