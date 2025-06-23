import re
import string
from collections import Counter
from typing import Tuple
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

if not hasattr(torch, "get_default_device"):

    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.get_default_device = get_default_device

model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_answer(s: str) -> str:
    """Normalize answer text by removing articles, punctuation, and whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Calculate F1 score between prediction and ground truth
    Returns tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """
    Calculate exact match between prediction and ground truth
    Returns boolean indicating if answers match exactly after normalization
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def cosine_similarity_score(prediction: str, ground_truth: str) -> float:
    embeddings = model.encode(
        [normalize_answer(prediction), normalize_answer(ground_truth)],
        convert_to_numpy=True,
    )
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
