from zbench.utils import load_jsonl, ndcg, load_json
from zbench.common_types import RerankerInput, Reranker, AnnotatedDataset, AnnotatedQueryDocuments, DatasetPairScoredPairs, Accuracy
import math
from zbench.rerankers import ZEROENTROPY_RERANKER, ZEROENTROPY_RERANKER_SMALL
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm
import asyncio

def visualize_ndcg_scores(ndcg_scores: list[float]) -> None:
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

async def benchmark_ndcg(annotation_path: str, reranker: Reranker, *, visualize: bool = False) -> list[float]:
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

async def benchmark_accuracy(ai_scores_path: str, reranker: Reranker) -> Accuracy:
    ai_scores = DatasetPairScoredPairs.model_validate(load_json(ai_scores_path))
    reranker_scores : list[list[float]] = await tqdm.gather(*[reranker(RerankerInput(query=scored_pair.pair.query, documents=[scored_pair.pair.document_a.content, scored_pair.pair.document_b.content])) for scored_pair in ai_scores.scored_pairs], desc="Reranking")
    num_correct = 0
    num_samples = 0
    num_consensus = 0
    num_consensus_correct = 0
    for scored_pair, reranker_score in zip(ai_scores.scored_pairs, reranker_scores, strict=True):
        ai_scores = [scored_pair.openai_score.score, scored_pair.gemini_score.score, scored_pair.anthropic_score.score]
        consensus : bool = ai_scores[0] * ai_scores[1] > 0 and ai_scores[1] * ai_scores[2] > 0
        reranker_value = reranker_score[1] - reranker_score[0]
        target_prefers_a : bool = reranker_value > 0
        target_prefers_b : bool = reranker_value < 0
        ai_value = sum([1 if x > 0 else (-1 if x < 0 else 0) for x in ai_scores])
        pred_prefers_a : bool = ai_value > 0
        pred_prefers_b : bool = ai_value < 0
        if (pred_prefers_a and target_prefers_a) or (pred_prefers_b and target_prefers_b):
            num_correct += 1
            if consensus:
                num_consensus_correct += 1
        num_samples += 1
        if consensus:
            num_consensus += 1
    return Accuracy(correct=num_correct, total=num_samples, consensus_correct=num_consensus_correct, consensus_total=num_consensus)

if __name__ == "__main__":
    asyncio.run(benchmark_ndcg("tmp/legalquad_annotated.jsonl", ZEROENTROPY_RERANKER_SMALL, visualize=True))
    asyncio.run(benchmark_accuracy("tmp/legalquad_ai_scores.json", ZEROENTROPY_RERANKER_SMALL))