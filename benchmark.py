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
    parser.add_argument('--num_samples', type=int, default=1, help='Number of questions to evaluate')
    parser.add_argument('--question_id', type=int, default=0, help='Add this argument if you want to check specific hotpot question')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    parser.add_argument('--mode', type=str, choices=['simple', 'standard', 'hybrid', 'adaptive', 'kg', 'rerank'], 
                        help="Version of QA chain")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == 'simple':
        qa_chain = SimpleQAChain()
    elif args.mode == 'adaptive':
        qa_chain = AdaptiveQAChain()
    else:
        qa_chain = QAChain(retrieval=args.mode)

    output_file = os.path.join(args.output_dir, f"rag_evaluation_{args.mode}.json")

    hotpotqa = load_hotpotqa_data(num_samples=args.num_samples, question_id=args.question_id)
    questions = hotpotqa['question']
    answers = hotpotqa['answer']
    supporting_sentences = []
    for facts, contex in zip(hotpotqa['supporting_facts'], hotpotqa['context']):
        supporting_sentences_for_question = []
        for title, sent_id in zip(facts['title'], facts['sent_id']):
            if title in contex['title']:
                idx = contex['title'].index(title)
                sentence = contex['sentences'][idx][sent_id]
                supporting_sentences_for_question.append(sentence)
        supporting_sentences.append(supporting_sentences_for_question) 

    evaluator = RAGEvaluator(qa_chain)
    print(f"Evaluating RAG pipeline with {args.num_samples} HotpotQA questions...")
    results, metrics = evaluator.evaluate(questions, answers, supporting_sentences, args.question_id, verbose=not args.quiet)
    evaluator.print_summary(metrics)

    # Load existing data if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            full_results = json.load(f)

        old_metrics = full_results["metrics"]
        old_count = old_metrics["total_questions"]
        new_count = metrics["total_questions"]
        total_count = old_count + new_count

        def weighted_avg(old_val, new_val):
            return (old_val * old_count + new_val * new_count) / total_count

        merged_metrics = {
            "total_questions": total_count,
            "total_time": old_metrics["total_time"] + metrics["total_time"],
            "avg_time_per_question": (old_metrics["total_time"] + metrics["total_time"]) / total_count,
            "avg_f1_score": weighted_avg(old_metrics["avg_f1_score"], metrics["avg_f1_score"]),
            "avg_exact_match": weighted_avg(old_metrics["avg_exact_match"], metrics["avg_exact_match"]),
            "avg_retrieval_precision": weighted_avg(old_metrics["avg_retrieval_precision"], metrics["avg_retrieval_precision"]),
            "avg_retrieval_recall": weighted_avg(old_metrics["avg_retrieval_recall"], metrics["avg_retrieval_recall"]),
            "avg_retrieval_f1": weighted_avg(old_metrics["avg_retrieval_f1"], metrics["avg_retrieval_f1"]),
            "question_type_counts": {}
        }

        merged_qtype_counts = old_metrics.get("question_type_counts", {})
        for k, v in metrics["question_type_counts"].items():
            merged_qtype_counts[k] = merged_qtype_counts.get(k, 0) + v
        merged_metrics["question_type_counts"] = merged_qtype_counts

        full_results["results"].extend(results)
        full_results["metrics"] = merged_metrics
        full_results["metadata"]["num_samples"] += args.num_samples
        full_results["metadata"]["last_appended"] = timestamp

    else:
        full_results = {
            "metadata": {
                "created": timestamp,
                "last_appended": timestamp,
                "algorithm_version": args.mode,
                "num_samples": args.num_samples
            },
            "results": results,
            "metrics": metrics
        }

    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults appended to '{output_file}'")


if __name__ == "__main__":
    main()