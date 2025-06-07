import argparse
from src.chains.qa_chain import QAChain
from src.chains.simple_qa_chain import SimpleQAChain
from src.chains.adaptive_qa_chain import AdaptiveQAChain


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Run a QA chain.")
    parser.add_argument("question", nargs="*", help="The question to ask")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "standard", "hybrid", "adaptive"],
        help="Version of QA chain",
    )
    args = parser.parse_args()

    # Initialize the QA chain
    if args.mode == "simple":
        qa_chain = SimpleQAChain()
    elif args.mode == "adaptive":
        qa_chain = AdaptiveQAChain()
    else:
        qa_chain = QAChain(retrieval=args.mode)

    # Get the question from command line arguments
    if args.question:
        question = " ".join(args.question)
        answer = qa_chain.run(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    else:
        print("Please provide a question as a command line argument.")
        print(
            "Example: python main.py 'Which house did Elizabeth I belong to?' --mode simple"
        )


if __name__ == "__main__":
    main()
