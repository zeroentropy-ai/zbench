from zeroentropy import AsyncZeroEntropy
from zbench.common_types import BaseReranker, RerankerInput
import asyncio
from zbench.common_types import AnnotatedDataset
from zbench.utils import load_jsonl
from typing import Literal

ZerankModels = Literal["zerank-1", "zerank-1-small"]

class Zerank(BaseReranker):
    def __init__(self, model: ZerankModels):
        self.model = model
        self.semaphore = asyncio.Semaphore(64)
        self.ze_client = AsyncZeroEntropy()
    
    async def score(self, input: RerankerInput) -> list[float]:
        async with self.semaphore:
            results = (await self.ze_client.models.rerank(query=input.query, documents=input.documents, model=self.model)).results
            results = sorted(results, key=lambda x: x.index)
            return [result.relevance_score for result in results]
