from zbench.utils import load_jsonl, ndcg
from zbench.common_types import RerankerInput, BaseReranker, Dataset, QueryDocuments, AnnotatedDataset, AnnotatedQueryDocuments
from zbench.rerankers import Zerank
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm
import asyncio
import math
from zbench.utils import argsort
import math

def _visualize_ndcg_scores(ndcg_scores: list[float]) -> None:
    """Visualize NDCG scores as a histogram."""
    plt.figure(figsize=(10, 6))
    plt.hist(ndcg_scores, bins=30, alpha=0.7, edgecolor='black', label='NDCG Scores')
    
    # Add mean line
    mean_score = sum(ndcg_scores) / len(ndcg_scores)
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
    
    plt.xlim(0, 1)
    plt.xlabel('NDCG Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of NDCG Scores')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# "score", 
# benchmark_ndcg(original_dataset_path: str, ground_truth_dataset_path: str)

def benchmark_ndcg(test_dataset_path: str, ground_truth_path: str, *, visualize: bool = False, document_limit: int = 10) -> list[float]:
    """Benchmark annotated dataset against ground truth annotated dataset using NDCG."""
    test_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(test_dataset_path)})
    ground_truth_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(ground_truth_path)})
    
    # Create mappings for both datasets by query ID and document ID
    test_scores = {}
    for test_data in test_dataset.data:
        test_scores[test_data.query.id] = {doc.id: doc.score for doc in test_data.documents}
        
    gt_scores = {}
    for gt_data in ground_truth_dataset.data:
        gt_scores[gt_data.query.id] = {doc.id: doc.score for doc in gt_data.documents}

    def calculate_ndcg(test_data) -> tuple[str, float]:
        query_id = test_data.query.id
        if query_id not in gt_scores:
            raise ValueError(f"Query ID {query_id} not found in ground truth dataset")
        
        # Limit documents if specified
        documents = test_data.documents[:document_limit] if document_limit else test_data.documents
        
        # Get test scores and ground truth scores, ensuring document ID matching
        test_doc_scores = []
        ground_truth_doc_scores = []
        
        for doc in documents:
            if doc.id not in gt_scores[query_id]:
                raise ValueError(f"Document ID {doc.id} not found for query {query_id} in ground truth dataset")
            if doc.id not in test_scores[query_id]:
                raise ValueError(f"Document ID {doc.id} not found for query {query_id} in test dataset")
                
            test_doc_scores.append(test_scores[query_id][doc.id])
            ground_truth_doc_scores.append(gt_scores[query_id][doc.id])
        
        # Convert ELO scores to relevance scores
        ground_truth_doc_scores = [math.exp(score) for score in ground_truth_doc_scores]
        
        return query_id, ndcg(ground_truth_doc_scores, test_doc_scores)
    
    ndcg_results = [calculate_ndcg(data) for data in test_dataset.data]
    ndcg_scores = [score for _, score in ndcg_results]

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

def benchmark_accuracy(test_dataset_path: str, ground_truth_path: str, *, document_limit: int = 10) -> list[float]:
    """Benchmark annotated dataset accuracy against ground truth annotated dataset."""
    test_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(test_dataset_path)})
    ground_truth_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(ground_truth_path)})
    
    # Create mappings for both datasets by query ID and document ID
    test_scores = {}
    for test_data in test_dataset.data:
        test_scores[test_data.query.id] = {doc.id: doc.score for doc in test_data.documents}
        
    gt_scores = {}
    for gt_data in ground_truth_dataset.data:
        gt_scores[gt_data.query.id] = {doc.id: doc.score for doc in gt_data.documents}

    def calculate_accuracy(test_data) -> tuple[str, float]:
        query_id = test_data.query.id
        if query_id not in gt_scores:
            raise ValueError(f"Query ID {query_id} not found in ground truth dataset")
        
        # Limit documents if specified
        documents = test_data.documents[:document_limit] if document_limit else test_data.documents
        
        # Get test scores and ground truth scores, ensuring document ID matching
        test_doc_scores = []
        ground_truth_doc_scores = []
        
        for doc in documents:
            if doc.id not in gt_scores[query_id]:
                raise ValueError(f"Document ID {doc.id} not found for query {query_id} in ground truth dataset")
            if doc.id not in test_scores[query_id]:
                raise ValueError(f"Document ID {doc.id} not found for query {query_id} in test dataset")
                
            test_doc_scores.append(test_scores[query_id][doc.id])
            ground_truth_doc_scores.append(gt_scores[query_id][doc.id])
        
        return query_id, _accuracy(test_doc_scores, ground_truth_doc_scores)

    accuracy_results = [calculate_accuracy(data) for data in test_dataset.data]
    accuracy_scores = [score for _, score in accuracy_results]

    return accuracy_scores

def recall_at_k(test_dataset_path: str, ground_truth_path: str, k: int, *, k_gt: int | None = None, document_limit: int = 10) -> list[float]:
    """Benchmark recall@k between two annotated datasets."""
    test_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(test_dataset_path)})
    ground_truth_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(ground_truth_path)})
    if k_gt is None:
        k_gt = k
    
    # Create mappings for both datasets by query ID and document ID
    test_scores = {}
    for test_data in test_dataset.data:
        test_scores[test_data.query.id] = {doc.id: doc.score for doc in test_data.documents}
        
    gt_scores = {}
    for gt_data in ground_truth_dataset.data:
        gt_scores[gt_data.query.id] = {doc.id: doc.score for doc in gt_data.documents}

    def calculate_recall(test_data) -> tuple[str, float]:
        query_id = test_data.query.id
        if query_id not in gt_scores:
            raise ValueError(f"Query ID {query_id} not found in ground truth dataset")
        
        # Limit documents if specified
        documents = test_data.documents[:document_limit] if document_limit else test_data.documents
        
        # Get test scores and ground truth scores, ensuring document ID matching
        test_doc_scores = []
        ground_truth_doc_scores = []
        
        for doc in documents:
            if doc.id not in gt_scores[query_id]:
                raise ValueError(f"Document ID {doc.id} not found for query {query_id} in ground truth dataset")
            if doc.id not in test_scores[query_id]:
                raise ValueError(f"Document ID {doc.id} not found for query {query_id} in test dataset")
                
            test_doc_scores.append(test_scores[query_id][doc.id])
            ground_truth_doc_scores.append(gt_scores[query_id][doc.id])
        
        truth_indices = argsort(ground_truth_doc_scores)[::-1][:k_gt]
        test_indices = argsort(test_doc_scores)[::-1][:k]
        intersections = set(truth_indices) & set(test_indices)
        return query_id, len(intersections) / k_gt
    
    recall_results = [calculate_recall(data) for data in test_dataset.data]
    recall_scores = [score for _, score in recall_results]
    return recall_scores

if __name__ == "__main__":
    # Example: Compare two annotated datasets
    benchmark_ndcg("tmp/test_annotated.jsonl", "tmp/ground_truth_annotated.jsonl", visualize=True)
    benchmark_accuracy("tmp/test_annotated.jsonl", "tmp/ground_truth_annotated.jsonl")
    recall_at_k("tmp/test_annotated.jsonl", "tmp/ground_truth_annotated.jsonl", 10, k_gt=10)