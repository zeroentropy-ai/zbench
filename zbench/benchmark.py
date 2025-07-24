from zbench.utils import load_jsonl, ndcg
from zbench.common_types import RerankerInput, BaseReranker, Dataset, QueryDocuments
from zbench.rerankers import Zerank, EnsembleReranker
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm
import asyncio
from zbench.utils import argsort
import math

def _visualize_ndcg_scores(ndcg_scores: list[float]) -> None:
    """Visualize NDCG scores as a histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(ndcg_scores, bins=30, alpha=0.7, edgecolor='black', label='NDCG Scores')
    
    # Add mean line
    mean_score = sum(ndcg_scores) / len(ndcg_scores)
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
    
    plt.xlabel('NDCG Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of NDCG Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

async def benchmark_ndcg(dataset_path: str, reranker: BaseReranker, ground_truth_reranker: BaseReranker, *, visualize: bool = False, document_limit: int = 10) -> list[float]:
    """Benchmark a reranker on an annotated dataset."""
    dataset = Dataset.model_validate({"data": load_jsonl(dataset_path)})

    async def calculate_ndcg(data: QueryDocuments) -> tuple[str, float]:
        reranker_scores = await reranker.score(RerankerInput(query=data.query.query, documents=[doc.content for doc in data.documents[:document_limit]]))
        ground_truth_scores = await ground_truth_reranker.score(RerankerInput(query=data.query.query, documents=[doc.content for doc in data.documents[:document_limit]]))
        ground_truth_scores = [math.exp(score) for score in ground_truth_scores]
        return data.query.id, ndcg(ground_truth_scores, reranker_scores)
    
    ndcg_scores : list[tuple[str, float]] = await tqdm.gather(*[calculate_ndcg(data) for data in dataset.data], desc="Calculating NDCG Scores")
    ndcg_scores = {query_id: score for query_id, score in ndcg_scores}
    ndcg_scores = [ndcg_scores[data.query.id] for data in dataset.data]

    if visualize:
        _visualize_ndcg_scores(ndcg_scores=ndcg_scores)

    return ndcg_scores

def _accuracy(reranker_scores: list[float], ground_truth_scores: list[float]) -> float:
    assert len(ground_truth_scores) == len(reranker_scores)
    n = len(ground_truth_scores)
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            reranker_value = reranker_scores[i] - reranker_scores[j]
            ground_truth_value = ground_truth_scores[i] - ground_truth_scores[j]
            if reranker_value * ground_truth_value > 0:
                correct += 1
    return (2 * correct) / (n * (n - 1))

async def benchmark_accuracy(dataset_path: str, reranker: BaseReranker, ground_truth_reranker: BaseReranker, *, document_limit: int = 10) -> list[float]:
    dataset = Dataset.model_validate({"data": load_jsonl(dataset_path)})    

    async def calculate_accuracy(data: QueryDocuments) -> tuple[str, float]:
        reranker_scores = await reranker.score(RerankerInput(query=data.query.query, documents=[doc.content for doc in data.documents[:document_limit]]))
        ground_truth_scores = await ground_truth_reranker.score(RerankerInput(query=data.query.query, documents=[doc.content for doc in data.documents[:document_limit]]))
        return data.query.id, _accuracy(reranker_scores, ground_truth_scores)

    accuracy_scores : list[tuple[str, float]] = await tqdm.gather(*[calculate_accuracy(data) for data in dataset.data], desc="Calculating Accuracy Scores")
    accuracy_scores = {query_id: score for query_id, score in accuracy_scores}
    accuracy_scores = [accuracy_scores[data.query.id] for data in dataset.data]

    return accuracy_scores

async def recall_at_k(dataset_path: str, reranker: BaseReranker, ground_truth_reranker: BaseReranker, k: int, *, k_gt: int | None = None, document_limit: int = 10) -> list[float]:
    dataset = Dataset.model_validate({"data": load_jsonl(dataset_path)})
    if k_gt is None:
        k_gt = k

    async def calculate_recall(data: QueryDocuments) -> tuple[str, float]:
        reranker_scores = await reranker.score(RerankerInput(query=data.query.query, documents=[doc.content for doc in data.documents[:document_limit]]))
        ground_truth_scores = await ground_truth_reranker.score(RerankerInput(query=data.query.query, documents=[doc.content for doc in data.documents[:document_limit]]))
        truth_indices = argsort(ground_truth_scores)[::-1][:k_gt]
        reranker_indices = argsort(reranker_scores)[::-1][:k]
        intersections = set(truth_indices) & set(reranker_indices)
        return data.query.id, len(intersections) / k_gt
    
    recall_scores : list[tuple[str, float]] = await tqdm.gather(*[calculate_recall(data) for data in dataset.data], desc="Calculating Recall Scores")
    recall_scores = {query_id: score for query_id, score in recall_scores}
    recall_scores = [recall_scores[data.query.id] for data in dataset.data]
    return recall_scores

if __name__ == "__main__":
    zerank = Zerank("zerank-1")
    reranker = EnsembleReranker("tmp/legalquad_annotated.jsonl")
    asyncio.run(benchmark_ndcg("tmp/legalquad.jsonl", zerank, reranker, visualize=True))
    asyncio.run(benchmark_accuracy("tmp/legalquad.jsonl", zerank, reranker))
    asyncio.run(recall_at_k("tmp/legalquad.jsonl", zerank, reranker, 10, k_gt=10))