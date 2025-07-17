from zeroentropy import AsyncZeroEntropy
from zbench.common_types import Reranker, RerankerInput
import asyncio

ze_client = AsyncZeroEntropy()
ze_semaphore = asyncio.Semaphore(64)

def zeroentropy_reranker_wrapper(model: str) -> Reranker:
    async def zeroentropy_reranker(input: RerankerInput) -> list[float]:
        async with ze_semaphore:
            results = (await ze_client.models.rerank(query=input.query, documents=input.documents, model=model)).results
            results = sorted(results, key=lambda x: x.index)
            return [result.relevance_score for result in results]
    return Reranker(reranker=zeroentropy_reranker)

ZEROENTROPY_RERANKER : Reranker = zeroentropy_reranker_wrapper("zerank-1")
ZEROENTROPY_RERANKER_SMALL : Reranker = zeroentropy_reranker_wrapper("zerank-1-small")
