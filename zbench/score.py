import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
import random
from tqdm.asyncio import tqdm_asyncio
import sys
import re
from zbench.utils import ROOT, wrap_sem
from zbench.ai import ai_call, AIModel, AIMessage, AIError
from zbench.common_types import DatasetPairDocument, DatasetPair, DatasetPairs, DatasetPairScore, DatasetPairScoredPair, DatasetPairScoredPairs

load_dotenv()

class Score:
    def __init__(self, pairs_path:str, scores_path:str)->None:
        self.pairs_path = pairs_path
        self.scores_path = scores_path
        self.sem = asyncio.Semaphore(150)
        random.seed("score")

    def load_pairs(self)->DatasetPairs:
        with open(self.pairs_path, "r") as f:
            pairs = DatasetPairs.model_validate_json(f.read())
        return pairs
    
    async def score_pair_structured(self,query: str, document_a: str, document_b: str, *, model: AIModel) -> DatasetPairScore:
        swap = random.random() < 0.5
        if swap:
            document_a, document_b = document_b, document_a

        class RelevanceScore(BaseModel):
            thoughts: list[str]
            score: float

        try:
            response = await ai_call(
                model=model,
                messages=[
                    AIMessage(
                        role="system",
                        content=f"""
# Task

You are a relevance scoring system. Given a query and two documents (A and B), your job is to decide which document is more relevant to the given query. You should think carefully, considering the pros and cons between each document. For your first few sentences, consider the pros and cons of Document A. Then, spend some time thinking about Document B. Then, at the end, compare, and make a decision as to which one is more relevant. Do NOT make a decision in the beginning of your thoughts, stay open-minded until the last 1-2 sentences of your thoughts.

# Scoring

The score should range from -1.0 to 1.0, where negative means document A is more relevant, and positive means Document B is more relevant.
You can pick any number from -1.0 to 1.0.
                        """,
                    ),
                    AIMessage(
                        role="user",
                        content=f"# Query:\n\n{query}\n\n# Document A:\n\n{document_a}\n\n# Document B:\n\n{document_b}\n\n",
                    )
                ],
                temperature=0,
                response_format=RelevanceScore,
            )
        except AIError as e:
            print("Unknown Exception!", e, file=sys.stderr)
            return DatasetPairScore(
                thought="",
                score=0.0,
            )
        thought = "\n".join(response.thoughts)
        score = response.score
        if swap:
            thought = f"(SWAPPED)\n{thought}"
            score = -score

        return DatasetPairScore(
            thought=thought,
            score=score,
        )

    async def score_pair_unstructured(self,query: str, document_a: str, document_b: str, *, model: AIModel) -> DatasetPairScore:
        swap = random.random() < 0.5
        if swap:
            document_a, document_b = document_b, document_a

        prev_thought = ""
        thought = ""
        score = 0.0
        try:
            messages=[
                AIMessage(
                    role="system",
                    content=f"""
# Task

You are a relevance scoring system. Given a query and two documents (A and B), your job is to decide which document is more relevant to the given query. You should think carefully, considering the pros and cons between each document. For your first few sentences, consider the pros and cons of Document A. Then, spend some time thinking about Document B. Then, at the end, compare, and make a decision as to which one is more relevant. Do NOT make a decision in the beginning of your thoughts, stay open-minded until the last 1-2 sentences of your thoughts. And, for the last 1-2 sentences, make a clear decision as to which document is more relevant. Ensure that by the last sentence of your thoughts that you've make a clear determination as to which document is more relevant, and also how strong that opinion is (e.g. slightly more relevant versus significantly more relevant).

# Scoring

The score should range from -1.0 to 1.0, where negative means document A is more relevant, and positive means Document B is more relevant.
You can pick any number from -1.0 to 1.0.

# Output Format

At the very end, your last line should be written in this format:
<score>
{{your_score:.2f}}
</score>

Of course, replacing your_score with a float between -1.0 and 1.0.
Do NOT output a score of 0.0, ensure to focus on which document is superior, and provide a negative or positive float between -1.0 and 1.0.
                    """,
                ),
                AIMessage(
                    role="user",
                    content=f"# Query:\n\n{query}\n\n# Document A:\n\n{document_a}\n\n# Document B:\n\n{document_b}\n\n",
                )
            ]
            for retry in range(2):
                response = await ai_call(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
                messages.append(AIMessage(role="assistant", content=response))

                re_result = re.search(r'<score>\s*([-+]?\d*\.\d+|\d+)\s*</score>', response)
                if re_result:
                    score = float(re_result.group(1))
                    thought = response.rsplit('<score>', 1)[0].strip()
                else:
                    score = 0.0
                    thought = response
                    if score == 0.0 and retry == 0:
                        prev_thought = thought
                        messages.append(AIMessage(role="user", content="You responded with a Score of 0.0. Please do NOT do this. You MUST output a score that is either negative, OR positive, but NOT 0.0. Please deeply consider whether or not Document A or Document B is preferrable. If, after thinking deeply, you still aren't sure, then just make your best guess."))
                        continue
                break
        except AIError as e:
            print("Unknown Exception!", e, file=sys.stderr)
            return DatasetPairScore(
                thought="",
                score=0.0,
            )
        
        if prev_thought != "":
            thought = f"Round 1: {prev_thought}\n\nRound 2: {thought}"

        if swap:
            thought = f"(SWAPPED)\n{thought}"
            score = -score

        return DatasetPairScore(
            thought=thought,
            score=score,
        )

    async def score_pair_ensemble(self,pair: DatasetPair) -> DatasetPairScoredPair:
        def format_document(document: DatasetPairDocument) -> str:
            return f"Metadata: {document.metadata}\nContent: {document.content}"
        query = pair.query
        document_a = format_document(pair.document_a)  
        document_b = format_document(pair.document_b)
        openai_score, gemini_score, anthropic_score = await asyncio.gather(
            self.score_pair_structured(
                query,
                document_a,
                document_b,
                model=AIModel(company="openai", model="gpt-4.1-2025-04-14"),
            ),
            self.score_pair_structured(
                query,
                document_a,
                document_b,
                model=AIModel(company="google", model="gemini-2.5-pro-preview-03-25"),
            ),
            self.score_pair_unstructured(
                query,  
                document_a,
                document_b,
                model=AIModel(company="anthropic", model="claude-3-7-sonnet-20250219"),
            )
        )
        return DatasetPairScoredPair(
            pair=pair,
            openai_score=openai_score,
            gemini_score=gemini_score,
            anthropic_score=anthropic_score,
        )
    
    async def score_pairs(self, pairs: DatasetPairs) -> DatasetPairScoredPairs:
        scored_pairs = await tqdm_asyncio.gather(
            *[
                wrap_sem(self.score_pair_ensemble(pair), self.sem)
                for pair in pairs.pairs
            ],
            desc="Scoring Pairs",
        )
        return DatasetPairScoredPairs(scored_pairs=scored_pairs)
    
    def save_scores(self, scores: DatasetPairScoredPairs) -> None:
        with open(self.scores_path, "w") as f:
            f.write(scores.model_dump_json(indent=4))
