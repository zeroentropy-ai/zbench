from zbench.utils import load_jsonl, ndcg
from zbench.common_types import RerankerInput, Reranker, AnnotatedDataset
import math
from zbench.rerankers import ZEROENTROPY_RERANKER
import numpy as np
import matplotlib.pyplot as plt

def visualize_ndcg_scores(ndcg_scores: list[float]) -> None:
    """Visualize NDCG scores as a histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(ndcg_scores, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('NDCG Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of NDCG Scores')
    plt.grid(True, alpha=0.3)
    plt.show()

def benchmark_reranker(annotation_path: str, reranker: Reranker, *, visualize: bool = False) -> float:
    """Benchmark a reranker on an annotated dataset."""
    annotated_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(annotation_path)})
    ndcg_scores = []

    for data in annotated_dataset.data:
        reranker_scores = reranker(RerankerInput(query=data.query.query, documents=[doc.content for doc in data.documents]))
        ground_truth_scores = [math.exp(doc.score) for doc in data.documents]
        ndcg_scores.append(ndcg(ground_truth_scores, reranker_scores))
    if visualize:
        visualize_ndcg_scores(ndcg_scores=ndcg_scores)
    return np.mean(ndcg_scores)

if __name__ == "__main__":
    benchmark_reranker("tmp/legalquad_annotated.jsonl", ZEROENTROPY_RERANKER, visualize=True)