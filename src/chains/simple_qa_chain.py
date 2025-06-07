from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config.settings import DEFAULT_LLM_MODEL, GROQ_API_URL, GROQ_API_KEY


class SimpleQAState(TypedDict):
    """State type for the simplified QA graph"""

    question: str
    answer: Optional[str]


class SimpleQAChain:
    """Simple question-answering chain using LangGraph without retrieval"""

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the Simple QA chain

        Args:
            model_name: Name of the LLM model to use
        """
        self.llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=0,
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_URL,
        )

        # Set up the answering template - no context needed
        qa_template = """Answer the following question directly and concisely:
        
Question: {question}

Be as simple and concise as possible — prefer short direct answers over full explanations.

Answer:"""
        self.qa_prompt = ChatPromptTemplate.from_template(qa_template)

        # Setup the graph
        self._setup_graph()

    def _setup_graph(self):
        """Set up the simplified QA graph with nodes and edges"""

        def generate_answer(state: SimpleQAState) -> Dict:
            """Generate an answer based directly on the question"""
            # Prepare the prompt with the question
            prompt = self.qa_prompt.invoke({"question": state["question"]})

            # Generate the answer
            answer = self.llm.invoke(prompt).content
            return {"answer": answer}

        # Create the graph
        graph = StateGraph(SimpleQAState)

        # Add the answer generation node
        graph.add_node("generation_node", generate_answer)

        # Connect to END
        graph.add_edge("generation_node", END)

        # Set the entry point
        graph.set_entry_point("generation_node")

        # Compile the graph
        self.graph = graph.compile()

    def invoke(self, input_dict: Dict[str, Any]) -> str:
        """
        Invoke the QA chain

        Args:
            input_dict: Input dictionary with 'question'

        Returns:
            Answer to the question
        """
        # Initialize state with input
        state = SimpleQAState(question=input_dict["question"], answer=None)

        # Run the graph
        result = self.graph.invoke(state)

        # Return the answer
        return result

    def run(self, question: str) -> str:
        """
        Run the QA chain with a question

        Args:
            question: User question

        Returns:
            Answer to the question
        """
        return self.invoke({"question": question})
