from zeroentropy import ZeroEntropy
from zbench.common_types import Reranker, RerankerInput

def zeroentropy_reranker(input: RerankerInput, model: str) -> list[float]:
    with ZeroEntropy() as ze_client:
        results = (ze_client.models.rerank(query=input.query, documents=input.documents, model=model)).results
    results = sorted(results, key=lambda x: x.index)
    return [result.relevance_score for result in results]


ZEROENTROPY_RERANKER : Reranker = Reranker(reranker=lambda input: zeroentropy_reranker(input, "zerank-1"))
ZEROENTROPY_RERANKER_SMALL : Reranker = Reranker(reranker=lambda input: zeroentropy_reranker(input, "zerank-1-small"))
