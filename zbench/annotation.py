import asyncio
from pathlib import Path
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm
import random
from shutil import copy2

from zbench.utils import (
    load_jsonl, load_json, save_pydantic_jsonl, save_pydantic_json, calculate_elos
)
from zbench.common_types import DatasetPairs, DatasetPair, Dataset, AnnotatedDataset, AnnotatedQueryDocuments, AnnotatedDocument, DatasetPairScoredPairs, DatasetPairDocument
from zbench.score import Score

class DatasetAnnotator:
    def __init__(self, dataset_path: str, *, cycle_num: int = 4, document_limit: int = 10):
        self.initial_path = Path(dataset_path)
        self.dataset_name = self.initial_path.stem
        self.working_dir = Path(f"data/annotation/{self.dataset_name}")
        self.initial_dir = Path(self.initial_path).parent
        self.cycle_num = cycle_num
        self.document_limit = document_limit
        
        # File paths
        self.dataset_path = self.working_dir / "dataset.jsonl"
        self.pairs_path = self.working_dir / "pairs.json"
        self.ai_scores_path = self.initial_dir / f"{self.dataset_name}_ai_scores.json"
        self.annotated_dataset_path = self.initial_dir / f"{self.dataset_name}_annotated.jsonl"
        
        # Ensure folder exists
        self.working_dir.mkdir(parents=True, exist_ok=True)
        copy2(self.initial_path, self.dataset_path)
    
    async def step1_load_dataset(self) -> Dataset:
        """Load dataset from the folder."""
        print(f"Step 1: Loading dataset")
        
        dataset : Dataset = Dataset.model_validate({"data": load_jsonl(str(self.dataset_path))})
        print(f"Loaded {len(dataset.data)} queries from {self.dataset_path}")
        return dataset
    
    def get_random_cycle_pairs(self, n : int) -> list[tuple[int, int]]:
        """Get random cycle pairs from the dataset."""
        values = list(range(n))
        random.shuffle(values)
        pairs = []
        for i in range(len(values)):
            pairs.append((values[i], values[(i + 1) % n]))
        return pairs

    def step2_create_pairs(self, dataset: Dataset) -> DatasetPairs:
        """Create pairs from the dataset."""
        print("Step 2: Creating pairs for AI scoring")
        
        pairs : DatasetPairs = DatasetPairs(pairs=[])
        for data in tqdm(dataset.data, desc="Creating pairs"):
            query = data.query
            documents = data.documents
            n = min(self.document_limit, len(data.documents))
            for _ in range(self.cycle_num):
                indices = self.get_random_cycle_pairs(n)
                for i, j in indices:
                    pairs.pairs.append(DatasetPair(
                        pair_id=f"{query.id}-{documents[i].id}-{documents[j].id}",
                        query_id=query.id,
                        query=query.query,
                        document_a=DatasetPairDocument(
                            document_id=documents[i].id,
                            metadata=documents[i].metadata,
                            content=documents[i].content,
                        ),
                        document_b=DatasetPairDocument(
                            document_id=documents[j].id,
                            metadata=documents[j].metadata,
                            content=documents[j].content,
                        ),
                    ))
            
        # Save pairs
        save_pydantic_json(self.pairs_path, pairs)
        print(f"Created {len(pairs.pairs)} pairs and saved to {self.pairs_path}")

        return pairs
    
    async def step3_score_pairs(self) -> None:
        """Score pairs."""
        print("Step 3: Scoring pairs")
        
        score = Score(pairs_path=str(self.pairs_path), scores_path=str(self.ai_scores_path))
        pairs = score.load_pairs()
        scores = await score.score_pairs(pairs)
        score.save_scores(scores)
        print(f"Scored {len(scores.scored_pairs)} pairs and saved to {self.ai_scores_path}")
    
    def step4_compose_annotated_dataset(self) -> None:
        """."""
        print("Step 4: Composing annotated dataset")

        # Load AI scores
        ai_scores : DatasetPairScoredPairs = DatasetPairScoredPairs.model_validate(load_json(str(self.ai_scores_path)))
        dataset : Dataset = Dataset.model_validate({"data": load_jsonl(str(self.dataset_path))})

        annotated_dataset : AnnotatedDataset = AnnotatedDataset(data=[])

        scores : dict[str, DatasetPairScoredPairs] = defaultdict(lambda: DatasetPairScoredPairs(scored_pairs=[]))
        for pair in ai_scores.scored_pairs:
            scores[pair.pair.query_id].scored_pairs.append(pair)
        for data in dataset.data:
            documents = data.documents
            id_to_idx = {doc.id: i for i, doc in enumerate(documents)}
            n = len(documents)
            w = np.zeros((n, n))
            for i in range(n):
                w[i,i] = 0.5
            for scored_pair in scores[data.query.id].scored_pairs:
                i = id_to_idx[scored_pair.pair.document_a.document_id]
                j = id_to_idx[scored_pair.pair.document_b.document_id]
                score = 0.0
                if scored_pair.openai_score.score > 0:
                    score += 1/3
                if scored_pair.anthropic_score.score > 0:
                    score += 1/3
                if scored_pair.gemini_score.score > 0:
                    score += 1/3
                w[j,i] += score
                w[i,j] += 1 - score
            elos, _ = calculate_elos(w)
            annotated_dataset.data.append(AnnotatedQueryDocuments(
                query=data.query,
                documents=[AnnotatedDocument(
                    id=documents[i].id,
                    content=documents[i].content,
                    metadata=documents[i].metadata,
                    score=elos[i]
                ) for i in range(n)]
            ))
        
        save_pydantic_jsonl(self.annotated_dataset_path, annotated_dataset.data)

        print(f"Created {len(annotated_dataset.data)} annotated lines")

        print(f"Saved annotated dataset to {self.annotated_dataset_path}")
    
    async def run_full_pipeline(self) -> None:
        """Run the complete pipeline."""
    
        try:
            # Step 1: Load data
            data = await self.step1_load_dataset()
            
            # Step 2: Create pairs
            pairs = self.step2_create_pairs(data)
            
            # Step 3: Score pairs
            await self.step3_score_pairs()

            # Step 4: Compose annotated dataset
            self.step4_compose_annotated_dataset()
            
            print(f"\n🎉 Annotation completed successfully!")
            print(f"Final outputs:")
            print(f"Annotated dataset: {self.annotated_dataset_path}")
            
        except Exception as e:
            print(f"❌ Annotation failed: {e}")
            raise

async def main():
    parser = argparse.ArgumentParser(description="Annotate dataset")
    parser.add_argument("dataset_path", help="Path to the dataset jsonl file")
    parser.add_argument("--cycle_num", type=int, default=4, help="Number of cycles to run")
    parser.add_argument("--document_limit", type=int, default=10, help="Number of documents to use for scoring")
    args = parser.parse_args()
    
    processor = DatasetAnnotator(
        dataset_path=args.dataset_path,
        cycle_num=args.cycle_num,
        document_limit=args.document_limit,
    )
    
    await processor.run_full_pipeline()


if __name__ == "__main__":
    asyncio.run(main()) 