from typing import Dict, Any, TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.retrieval.retriever import RerankRetriever, StandardRetriever
from src.config.settings import DEFAULT_LLM_MODEL, GROQ_API_KEY, GROQ_API_URL


class QAState(TypedDict):
    """State type for the QA graph"""

    question: str
    question_type: Optional[str]
    context: Optional[str]
    answer: Optional[str]


class AdaptiveQAChain:
    """
    Question-answering chain using LangGraph with intelligent routing
    between no RAG, standard RAG, and KG RAG approaches based on question complexity
    """

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the QA chain
        Args:
            model_name: Name of the LLM model to use
        """
        self.llm = ChatOpenAI(
            model=DEFAULT_LLM_MODEL,
            temperature=0,
            api_key=GROQ_API_KEY,
            base_url=GROQ_API_URL,
        )

        # Initialize retrievers
        self.standard_retriever = StandardRetriever()
        self.kg_retriever = RerankRetriever(self.llm)

        # Set up the router template
        router_template = """You need to determine the best approach to answer the following question:
Question: {question}

Choose exactly ONE of the following approaches:
1. "no_rag" - For simple questions that can be answered with the model's internal knowledge (general knowledge, facts, common concepts)
2. "standard_rag" - For questions needing specific information retrieval but minimal relationship analysis
3. "kg_rag" - For questions needing retrieval involving entity relationships or connections between concepts.

Reply with just one of these options (no_rag, standard_rag, or kg_rag) based on which approach would best answer the question.
"""
        self.router_prompt = ChatPromptTemplate.from_template(router_template)

        # Set up the answering template with context
        qa_with_context_template = """Answer the question based only on the following context:
{context}

Question: {question}

Be as simple and concise as possible — prefer short direct answers over full explanations.
Answer:"""
        self.qa_with_context_prompt = ChatPromptTemplate.from_template(
            qa_with_context_template
        )

        # Set up the answering template without context (for no_rag)
        qa_no_context_template = """Answer the following question using your general knowledge:

Question: {question}

Be as simple and concise as possible — prefer short direct answers over full explanations.
Answer:"""
        self.qa_no_context_prompt = ChatPromptTemplate.from_template(
            qa_no_context_template
        )

        # Setup the graph
        self._setup_graph()

    def _setup_graph(self):
        """Set up the QA graph with nodes and edges for routing between approaches"""

        def route_question(state: QAState) -> Dict:
            """Determine which retrieval approach to use"""
            # Prepare the prompt
            prompt = self.router_prompt.invoke({"question": state["question"]})

            # Get the routing decision
            question_type = self.llm.invoke(prompt).content.strip().lower()

            # Normalize to one of our expected values
            if "no_rag" in question_type:
                question_type = "no_rag"
            elif "kg_rag" in question_type:
                question_type = "kg_rag"
            else:
                question_type = "standard_rag"

            return {"question_type": question_type}

        def retrieve_standard(state: QAState) -> Dict:
            """Retrieve relevant documents using standard retriever"""
            context = self.standard_retriever.retrieve(state["question"])
            return {"context": context}

        def retrieve_kg(state: QAState) -> Dict:
            """Retrieve relevant documents using KG retriever with knowledge graph"""
            context = self.kg_retriever.retrieve(state["question"])
            return {"context": context}

        def generate_answer_with_context(state: QAState) -> Dict:
            """Generate an answer based on the context and question"""
            # Prepare the prompt with context
            prompt = self.qa_with_context_prompt.invoke(
                {"context": state["context"], "question": state["question"]}
            )

            # Generate the answer
            answer = self.llm.invoke(prompt).content
            return {"answer": answer}

        def generate_answer_no_context(state: QAState) -> Dict:
            """Generate an answer using only the model's knowledge"""
            # Prepare the prompt without context
            prompt = self.qa_no_context_prompt.invoke({"question": state["question"]})

            # Generate the answer
            answer = self.llm.invoke(prompt).content
            return {"answer": answer}

        # Define the conditional routing function
        def route_by_question_type(
            state: QAState,
        ) -> Literal["standard_retrieval", "kg_retrieval", "no_retrieval"]:
            """Route to the appropriate retrieval method based on question type"""
            if state["question_type"] == "kg_rag":
                return "kg_retrieval"
            elif state["question_type"] == "standard_rag":
                return "standard_retrieval"
            else:  # no_rag
                return "no_retrieval"

        # Create the graph
        graph = StateGraph(QAState)

        # Add nodes
        graph.add_node("router", route_question)
        graph.add_node("standard_retrieval", retrieve_standard)
        graph.add_node("kg_retrieval", retrieve_kg)
        graph.add_node(
            "no_retrieval",
            lambda x: {"context": "No context provided - using model knowledge"},
        )
        graph.add_node("answer_with_context", generate_answer_with_context)
        graph.add_node("answer_no_context", generate_answer_no_context)

        # Add edges with conditional routing
        graph.add_conditional_edges(
            "router",
            route_by_question_type,
            {
                "standard_retrieval": "standard_retrieval",
                "kg_retrieval": "kg_retrieval",
                "no_retrieval": "no_retrieval",
            },
        )

        # Connect retrieval nodes to appropriate answer generators
        graph.add_edge("standard_retrieval", "answer_with_context")
        graph.add_edge("kg_retrieval", "answer_with_context")
        graph.add_edge("no_retrieval", "answer_no_context")

        # Connect answer generators to end
        graph.add_edge("answer_with_context", END)
        graph.add_edge("answer_no_context", END)

        # Set the entry point
        graph.set_entry_point("router")

        # Compile the graph
        self.graph = graph.compile()

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Initialize state with input
        state = QAState(
            question=input_dict["question"],
            question_type=None,
            context=None,
            answer=None,
        )

        # Run the graph
        result = self.graph.invoke(state)

        # Return the result with metadata
        return {
            "answer": result["answer"],
            "question_type": result["question_type"],
            "used_context": result["context"]
            != "No context provided - using model knowledge",
        }

    def run(self, question: str) -> str:
        result = self.invoke({"question": question})
        return result
