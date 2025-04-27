from src.chains.qa_chain import QAChain
from typing import List, Tuple

def interactive_qa():
    """Run an interactive QA session"""
    print("Loading QA system...")
    qa_chain = QAChain()
    
    print("\nWelcome to the Knowledge Graph QA System!")
    print("Type 'exit' or 'quit' to end the session.")
    
    chat_history = []
    
    while True:
        # Get user input
        question = input("\nYour question: ")
        
        # Check if the user wants to exit
        if question.lower() in ['exit', 'quit']:
            print("Thank you for using the Knowledge Graph QA System. Goodbye!")
            break
        
        # Get the answer
        answer = qa_chain.run(question, chat_history)
        
        # Print the answer
        print(f"\nAnswer: {answer}")
        
        # Update chat history
        chat_history.append((question, answer))
        
        # Limit chat history to last 5 conversations to prevent context overflow
        if len(chat_history) > 5:
            chat_history = chat_history[-5:]

if __name__ == "__main__":
    interactive_qa()