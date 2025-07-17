from zbench.utils import load_jsonl, ndcg
from zbench.common_types import RerankerInput, Reranker, AnnotatedDataset, AnnotatedQueryDocuments
import math
from zbench.rerankers import ZEROENTROPY_RERANKER
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm
import asyncio

def visualize_ndcg_scores(ndcg_scores: list[float]) -> None:
    """Visualize NDCG scores as a histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(ndcg_scores, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('NDCG Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of NDCG Scores')
    plt.grid(True, alpha=0.3)
    plt.show()

async def benchmark_reranker(annotation_path: str, reranker: Reranker, *, visualize: bool = False) -> list[float]:
    """Benchmark a reranker on an annotated dataset."""
    annotated_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(annotation_path)})

    async def calculate_ndcg(data: AnnotatedQueryDocuments) -> tuple[str, float]:
        reranker_scores = await reranker(RerankerInput(query=data.query.query, documents=[doc.content for doc in data.documents]))
        ground_truth_scores = [math.exp(doc.score) for doc in data.documents]
        return data.query.id, ndcg(ground_truth_scores, reranker_scores)
    
    ndcg_scores : list[tuple[str, float]] = await tqdm.gather(*[calculate_ndcg(data) for data in annotated_dataset.data], desc="Calculating NDCG Scores")
    ndcg_scores = {query_id: score for query_id, score in ndcg_scores}
    ndcg_scores = [ndcg_scores[data.query.id] for data in annotated_dataset.data]

    if visualize:
        visualize_ndcg_scores(ndcg_scores=ndcg_scores)

    return ndcg_scores

if __name__ == "__main__":
    asyncio.run(benchmark_reranker("tmp/legalquad_annotated.jsonl", ZEROENTROPY_RERANKER, visualize=True))