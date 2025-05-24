import json
import argparse
import datetime
import os
from src.data.eval_loaders import load_hotpotqa_data
from src.evaluation.evaluator import RAGEvaluator
from src.chains.qa_chain import QAChain
from src.chains.simple_qa_chain import SimpleQAChain
from src.chains.adaptive_qa_chain import AdaptiveQAChain

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG pipeline with HotpotQA')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of questions to evaluate')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    parser.add_argument('--mode', type=str, choices=['simple', 'standard', 'hybrid', 'adaptive'], 
                        help="Version of QA chain")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get current timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine which algorithm version to use
    if args.mode == 'simple':
        qa_chain = SimpleQAChain()
    elif args.mode == 'adaptive':
        qa_chain = AdaptiveQAChain()
    else:
        qa_chain = QAChain(retrieval=args.mode)
    
    # Create unique output filename with timestamp and algorithm version
    output_file = os.path.join(
        args.output_dir, 
        f"rag_evaluation_{args.mode}_{timestamp}.json"
    )
    
    hotpotqa = load_hotpotqa_data(num_samples=args.num_samples)
    questions = hotpotqa['question']
    answers = hotpotqa['answer']
    
    print(questions)
    print(answers)
    
    evaluator = RAGEvaluator(qa_chain)
    print(f"Evaluating RAG pipeline with {args.num_samples} HotpotQA questions...")
    results, metrics = evaluator.evaluate(questions, answers, verbose=not args.quiet)
    evaluator.print_summary(metrics)
    
    # Add metadata to results
    full_results = {
        "metadata": {
            "timestamp": timestamp,
            "algorithm_version": args.mode,
            "num_samples": args.num_samples
        },
        "results": results,
        "metrics": metrics
    }
    
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nDetailed results saved to '{output_file}'")

if __name__ == "__main__":
    main()