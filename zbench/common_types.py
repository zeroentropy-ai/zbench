from pydantic import BaseModel
from typing import Any
from abc import ABC, abstractmethod
from zbench.utils import load_jsonl, save_pydantic_jsonl
    
class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any] = {}

class AnnotatedDocument(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any] = {}
    score: float

class Query(BaseModel):
    id: str
    query: str

class QueryDocuments(BaseModel):
    query: Query
    documents: list[Document]

class AnnotatedQueryDocuments(BaseModel):
    query: Query
    documents: list[AnnotatedDocument]

class Dataset(BaseModel):
    data: list[QueryDocuments]

class AnnotatedDataset(BaseModel):
    data: list[AnnotatedQueryDocuments]

# Pairs Dataset

class DatasetPairDocument(BaseModel):
    document_id: str
    metadata: dict[str, Any]
    content: str


class DatasetPair(BaseModel):
    pair_id: str
    query_id: str
    query: str
    document_a: DatasetPairDocument
    document_b: DatasetPairDocument


class DatasetPairs(BaseModel):
    pairs: list[DatasetPair]


# Pairs Scores

class DatasetPairScore(BaseModel):
    thought: str
    score: float

class ModelScore(BaseModel):
                model: str
                status: str
                scores: list[DatasetPairScore]

class DatasetPairScoredPair(BaseModel):
    pair: DatasetPair
    openai_score: DatasetPairScore
    gemini_score: DatasetPairScore
    anthropic_score: DatasetPairScore


class DatasetPairScoredPairs(BaseModel):
    scored_pairs: list[DatasetPairScoredPair]

class RerankerInput(BaseModel):
    query: str
    documents: list[str]

class BaseReranker(ABC):
    @abstractmethod
    async def score(self, input: RerankerInput) -> list[float]:
        pass
    async def annotate(self, dataset_path: str, annotated_dataset_path: str, *, document_limit: int | None = None) -> None:
        """Annotate a dataset using this reranker and save as annotated dataset."""
        dataset = Dataset.model_validate({"data": load_jsonl(dataset_path)})
        annotated_data : list[AnnotatedQueryDocuments] = []
        
        for data in dataset.data:
            # Limit documents if specified
            documents = data.documents[:document_limit] if document_limit else data.documents
            
            # Get scores from reranker
            reranker_scores = await self.score(RerankerInput(
                query=data.query.query, 
                documents=[doc.content for doc in documents]
            ))
            
            # Create annotated documents
            annotated_documents = [
                AnnotatedDocument(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score
                )
                for doc, score in zip(documents, reranker_scores)
            ]
            
            # Create annotated query documents
            annotated_query_docs = AnnotatedQueryDocuments(
                query=data.query,
                documents=annotated_documents
            )
            annotated_data.append(annotated_query_docs)
        
        # Save annotated dataset
        save_pydantic_jsonl(annotated_dataset_path, annotated_data)