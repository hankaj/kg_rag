from typing import List, Dict, Any
from datasets import load_dataset

def load_hotpotqa_data(num_samples: int = 3) -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset from Hugging Face datasets
    Returns list of question/answer pairs
    """
    hotpotqa_dataset = load_dataset("hotpot_qa", "distractor")
    hotpotqa_questions = hotpotqa_dataset["train"]
    print(f"Successfully loaded HotpotQA with {len(hotpotqa_questions)} questions in validation set.")
    
    return hotpotqa_questions[:num_samples]
