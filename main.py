from src.chains.qa_chain import QAChain
import sys

def main():
    """Main entry point for the application"""
    # Initialize the QA chain
    qa_chain = QAChain()
    
    # Get the question from command line arguments
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        # Run the QA chain with the question
        answer = qa_chain.run(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    else:
        print("Please provide a question as a command line argument.")
        print("Example: python main.py 'Which house did Elizabeth I belong to?'")

if __name__ == "__main__":
    main()