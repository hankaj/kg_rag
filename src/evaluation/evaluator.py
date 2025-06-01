import time
from typing import Dict, List, Any, Tuple
from collections import Counter
from tqdm import tqdm

from src.evaluation.metrics import f1_score, exact_match_score


def calculate_retrieval_metrics(retrieved_facts: List[str], supporting_sentences: List[str]) -> Tuple[float, float, float]:
    if not retrieved_facts or not supporting_sentences:
        return 0.0, 0.0, 0.0

    relevant_retrieved = 0
    for sentence in supporting_sentences:
        for fact in retrieved_facts:
            if sentence.lower() in fact.lower():
                relevant_retrieved += 1
                break
    
    precision = relevant_retrieved / len(retrieved_facts)
    recall = relevant_retrieved / len(supporting_sentences)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

class RAGEvaluator:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
    
    def evaluate(self, questions: List[str], answers: List[str], supporting_sentences: List[str], verbose: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:

        results = []
        start_time = time.time()
        
        total_f1 = 0.0
        total_em = 0
        total_retrieval_precision = 0.0
        total_retrieval_recall = 0.0
        total_retrieval_f1 = 0.0
        
        for i, (question, reference_answer, supporting_facts) in enumerate(tqdm(zip(questions, answers, supporting_sentences))):
            
            try:
                result = self.qa_chain.run(question)
                answer = result['answer']
                question_type = result.get("question_type", None)
                
                retrieved_docs = result['context']
                
                # Calculate retrieval metrics
                retrieval_precision, retrieval_recall, retrieval_f1 = calculate_retrieval_metrics(
                    retrieved_docs, supporting_facts
                )
                
                f1, precision, recall = f1_score(answer, reference_answer)
                em = exact_match_score(answer, reference_answer)
                
                total_f1 += f1
                total_em += int(em)
                total_retrieval_precision += retrieval_precision
                total_retrieval_recall += retrieval_recall
                total_retrieval_f1 += retrieval_f1
                
                results.append({
                    "question_id": i,
                    "question": question,
                    "reference_answer": reference_answer,
                    "model_answer": answer,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "exact_match": em,
                    "retrieval_precision": retrieval_precision,
                    "retrieval_recall": retrieval_recall,
                    "retrieval_f1": retrieval_f1,
                    "question_type": question_type
                })
                
                if verbose:
                    print(f"\nQuestion {i+1}: {question}")
                    print(f"Reference Answer: {reference_answer}")
                    print(f"Model Answer: {answer}")
                    print(f"F1 Score: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
                    print(f"Exact Match: {em}")
                    print(f"Retrieval F1 Score: {retrieval_f1:.4f} (Precision: {retrieval_precision:.4f}, Recall: {retrieval_recall:.4f})")
                    if question_type:
                        print(f"Question Type: {question_type}")
                    
                    print("-" * 80)
                
            except Exception as e:
                print(f"\nError processing question {i+1}: {str(e)}")
                break
                    
        
        total_time = time.time() - start_time
        
        num_successful = len(results)
        avg_time_per_question = total_time / len(results) if results else 0
        avg_f1 = total_f1 / num_successful if num_successful > 0 else 0
        avg_em = total_em / num_successful if num_successful > 0 else 0
        avg_retrieval_precision = total_retrieval_precision / num_successful if num_successful > 0 else 0
        avg_retrieval_recall = total_retrieval_recall / num_successful if num_successful > 0 else 0
        avg_retrieval_f1 = total_retrieval_f1 / num_successful if num_successful > 0 else 0

        question_type_counts = Counter(
            r["question_type"] for r in results if r["question_type"]
        )
        
        metrics = {
            "total_questions": len(results),
            "total_time": total_time,
            "avg_time_per_question": avg_time_per_question,
            "avg_f1_score": avg_f1,
            "avg_exact_match": avg_em,
            "avg_retrieval_precision": avg_retrieval_precision,
            "avg_retrieval_recall": avg_retrieval_recall,
            "avg_retrieval_f1": avg_retrieval_f1,
            "question_type_counts": dict(question_type_counts)
        }
        
        return results, metrics
    
    def print_summary(self, metrics: Dict[str, float]) -> None:
        """Print summary of evaluation metrics"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS:")
        print(f"Total questions processed: {metrics['total_questions']}")
        print(f"Average F1 score: {metrics['avg_f1_score']:.4f}")
        print(f"Average Retrieval F1 score: {metrics['avg_retrieval_f1']:.4f}")
        print(f"Average Exact Match score: {metrics['avg_exact_match']:.4f}")
        print(f"Total processing time: {metrics['total_time']:.2f} seconds")
        print(f"Average time per question: {metrics['avg_time_per_question']:.2f} seconds"),
        print(f"Count of different paths: {metrics['question_type_counts']}")
