from pydantic import BaseModel
from typing import Any
from abc import ABC, abstractmethod

class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]

class AnnotatedDocument(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]
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