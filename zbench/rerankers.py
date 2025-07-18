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
        
class EnsembleReranker(BaseReranker):
    def __init__(self, annotation_path: str):
        self.annotation_path = annotation_path
        annotated_dataset = AnnotatedDataset.model_validate({"data": load_jsonl(annotation_path)})
        self.query_data : dict[str, dict[str, float]] = {data.query.query: {doc.content: doc.score for doc in data.documents} for data in annotated_dataset.data}
    
    async def score(self, input: RerankerInput) -> list[float]:
        if input.query not in self.query_data:
            raise ValueError(f"Query {input.query} not found in annotated dataset {self.annotation_path}")
        doc_scores = self.query_data[input.query]
        for document in input.documents:
            if document not in doc_scores:
                raise ValueError(f"Document {document} not found for query {input.query} in annotated dataset {self.annotation_path}")
        return [doc_scores[document] for document in input.documents]
