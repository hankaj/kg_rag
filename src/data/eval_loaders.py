from typing import List, Dict, Any
from datasets import load_dataset


def load_hotpotqa_data(num_samples: int = 3, question_id=0) -> List[Dict[str, Any]]:
    hotpotqa_dataset = load_dataset("hotpot_qa", "distractor")
    hotpotqa_questions = hotpotqa_dataset["train"][
        question_id : question_id + num_samples
    ]
    print(
        f"Successfully loaded HotpotQA with {len(hotpotqa_questions)} questions in validation set."
    )
    return hotpotqa_questions
