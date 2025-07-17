# zbench

**zbench** is a comprehensive platform for annotating query-to-document relevancy and backtesting performance of custom rerankers. It uses an ensemble of state-of-the-art LLM models to generate high-quality annotations using ELO-like rating system and provides tools for evaluating reranker performance using NDCG metric.

## Features

- **AI-Powered Annotation**: Uses ensemble scoring with OpenAI GPT-4, Anthropic Claude, and Google Gemini models
- **Pairwise Comparison**: Generates pairwise document-to-document comparisons and converts them to ELO ratings
- **Custom Reranker Support**: Easy integration of custom rerankers for benchmarking
- **Comprehensive Evaluation**: NDCG scoring and visualization tools

### Environment Setup

For annotation purposes, zbench is going to call an ensemble of OpenAI GPT-4, Anthropic Claude, and Google Gemini. To make this work, create a `.env` file in the root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Approximate price of annotation: 20 USD / 1000 pairwise comparisons (Note: one query annotation takes around `(cycle_num) * (number of documents for the query)` comparisons)

## Quick Start

### 1. Annotate a Dataset

```bash
python zbench.annotation path/to/your/dataset.jsonl
```

### 2. Benchmark a Reranker

```python
import asyncio
from zbench.benchmark import benchmark_reranker
from zbench.rerankers import ZEROENTROPY_RERANKER

async def main():
    scores = await benchmark_reranker(
        "path/to/annotated_dataset.jsonl", 
        ZEROENTROPY_RERANKER,
        visualize=True
    )
    print(f"Average NDCG: {sum(scores) / len(scores):.4f}")

asyncio.run(main())
```

## Data Formats

### Input Dataset Format

Your input dataset should be a JSONL file where each line is in `zbench.common_types.QueryDocuments` format:

```json
{
  "query": {
    "id": "query_001",
    "query": "What are the benefits of renewable energy?"
  },
  "documents": [
    {
      "id": "doc_001",
      "content": "Solar power is a clean, renewable energy source...",
      "metadata": {
        "source": "energy_report.pdf",
        "page": 15
      }
    },
    {
      "id": "doc_002", 
      "content": "Wind energy has become increasingly cost-effective...",
      "metadata": {
        "source": "wind_study.pdf",
        "page": 3
      }
    }
  ]
}
```

**Required Fields:**
- `query.id`: Unique identifier for the query
- `query.query`: The actual query text
- `documents[].id`: Unique identifier for each document
- `documents[].content`: The document content/text

### Annotated Dataset Format

The annotation process produces a JSONL file named the same way as an input but with "_annotated" suffix. Each line contains scored documents in the format of `zbench.common_types.AnnotatedQueryDocuments`:

```json
{
  "query": {
    "id": "query_001",
    "query": "What are the benefits of renewable energy?"
  },
  "documents": [
    {
      "id": "doc_001",
      "content": "Solar power is a clean, renewable energy source...",
      "metadata": {
        "source": "energy_report.pdf",
        "page": 15
      },
      "score": 2.45
    },
    {
      "id": "doc_002",
      "content": "Wind energy has become increasingly cost-effective...", 
      "metadata": {
        "source": "wind_study.pdf",
        "page": 3
      },
      "score": 1.87
    }
  ]
}
```

**New Fields:**
- `documents[].score`: ELO-based relevance score (higher = more relevant)

## Annotation Pipeline

The annotation process consists of four main steps:

### Step 1: Load Dataset
Loads and validates the input dataset format.

### Step 2: Create Pairs
Generates pairwise comparisons between documents for each query using a random cycle approach. This ensures comprehensive coverage while managing computational cost.

**Configuration Options:**
- `--cycle_num`: Number of random cycles for pair generation (default: 4)
- `--document_threshold`: Maximum number of documents to use per query (default: 10). Uses the first `--document_threshold` documents in the input order

### Step 3: Score Pairs
Uses an ensemble of three AI models to score each document pair:
- **OpenAI GPT-4**
- **Google Gemini**
- **Anthropic Claude**

Each model scores pairs on a scale from -1.0 to 1.0, where:
- Negative values indicate Document A is more relevant
- Positive values indicate Document B is more relevant

In addition, each model provides a human-readable reasoning along with the scores. `(SWAPPED)` at the beginning of a reasoning means that the model got the document in an inverse order (`document_a` is `document_b` and vice versa). This swap was done for uniformly random pairs in order to make annotations unbiased of the ordering. All of the scoring data are stored in the `/data/annotation/{input file name}/ai_scores.json` using `zbench.common_types.DatasetPairScoredPairs` format.

### Step 4: Compose Annotated Dataset
Converts pairwise scores into ELO ratings using an algorithm from https://hackmd.io/@-Gjw1zWMSH6lMPRlziQFEw/B15B4Rsleg. The final scores represent relative document relevance within each query.

## Custom Rerankers

### Defining a Custom Reranker

A reranker is a `zbench.common_types.Reranker` class that contains an async function that takes `zbench.common_types.RerankerInput` (a query and list of documents) and returns relevance scores for all of them:

```python
from zbench.common_types import Reranker, RerankerInput

async def my_custom_reranker(input: RerankerInput) -> list[float]:
    """
    Custom reranker implementation.
    
    Args:
        input: Contains query (str) and documents (list[str])
        
    Returns:
        List of relevance scores (higher = more relevant)
    """
    query = input.query
    documents = input.documents
    
    # Your reranking logic here
    scores = []
    for doc in documents:
        # Example: simple keyword matching
        score = sum(1 for word in query.lower().split() 
                   if word in doc.lower())
        scores.append(float(score))
    
    return scores

# Wrap your function in a Reranker object
MY_RERANKER : Reranker = Reranker(reranker=my_custom_reranker)
```

Note: Custom rerankers are called simultaneously for every query, so make sure to utilize `asyncio.Semaphore` to avoid ratelimit issues.

### ZeroEntropy Rerankers

zbench includes built-in ZeroEntropy rerankers:

```python
from zbench.rerankers import ZEROENTROPY_RERANKER, ZEROENTROPY_RERANKER_SMALL

# Use the full model
scores = await benchmark_reranker("dataset.jsonl", ZEROENTROPY_RERANKER)

# Use the small model
scores = await benchmark_reranker("dataset.jsonl", ZEROENTROPY_RERANKER_SMALL)
```

## Evaluation and Benchmarking

### NDCG Calculation

zbench uses Normalized Discounted Cumulative Gain (NDCG) to evaluate reranker performance:

```python
from zbench.utils import ndcg

# Ground truth relevance scores = e ^ (corresponding elo)
ground_truth = [3.2, 2.1, 1.8, 0.9]  # Higher = more relevant

# Predicted relevance scores from your reranker
predictions = [2.8, 2.3, 1.2, 1.0]

# Calculate NDCG
score = ndcg(ground_truth, predictions)
print(f"NDCG: {score:.4f}")
```

### Visualization

Enable visualization to see NDCG score distributions:

```python
scores = await benchmark_reranker(
    "annotated_dataset.jsonl", 
    MY_RERANKER,
    visualize=True  # Shows histogram of NDCG scores
)
```

### Working with Large Datasets

For large datasets, consider:

1. **Increase document threshold gradually**: Start with 10, increase if needed
2. **Use fewer cycles for initial testing**: In practice, no more than 4 cycles are needed for the ELO convergence, but you can lower to 2-3 for small document samples
3. **Monitor API costs**: Each pair requires 3 AI model calls, approximate cost: 20 USD / 1000 pairwise comparisons
4. **Implement checkpointing**: Save intermediate results, do not run on thousands of queries right away
