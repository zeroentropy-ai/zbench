# zbench

> **zELO Method**: For a given query and K candidate documents, use an Ensemble of large LLMs to annotate pairwise "battles" between candidate documents. Then, calculate ELO scores for all of the documents via [bradley-terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model). We call this final score a **zELO score**.

**zbench** is a comprehensive platform for annotating query-to-document relevancy and backtesting performance of custom rerankers and initial retrieval methods. It uses an ensemble of state-of-the-art LLM models to generate high-quality annotations using the **zELO** rating system, and provides tools for evaluating retriever performance using NDCG and recall metrics.

## Features

- **AI-Powered Annotation**: Uses ensemble scoring with OpenAI GPT-4, Anthropic Claude, and Google Gemini models
- **Pairwise Comparison**: Generates pairwise document-to-document comparisons and converts them to zELO ratings
- **Custom Reranker Support**: Easy integration of custom rerankers for benchmarking
- **Comprehensive Evaluation**: NDCG scoring and visualization tools

## Thesis

When a single LLM is given a pair of documents d1, d2, and is given the task of deciding which document is more relevant to a query q, we can prompt engineer the LLM until we find a fairly uniform distribution between -1 and 1 (Where -1 indicates a preference of d1, and 1 indicates a preference of d2). However, often a single LLM is quite noisy, and an annotation of -1 or 1 isn't a strong indication by itself.

However - when we inference three LLMs, _consensus_ between the LLMs becomes an extremely strong indicator of fundamental relevancy. In-fact, when we ran numerous double-blind annotations, LLM consensus is associated with a >97% probability that trusted high-quality human annotators would prefer that document as well. Most retrieval systems only align 60-70% of the time, showing a large gap between existing retrieval systems and high-quality human annotations.

By taking LLM-annotated pairwise comparisons, where red is a negative number and green is a positive number, we can plot the full KxK matrix of comparisons.

<img width="734" height="259" alt="The pairwise comparison matrix, and distribution of ELO scores" src="https://github.com/user-attachments/assets/12717699-2fda-4676-be75-2aa089cac3c0" />

> Graphs showing pairwise comparisons. The color at (i, j) / (j, i) is based on inferencing the Ensemble of LLMs for d_i and d_j. The first matrix is when the indices are sorted by hybrid search. Horizontal lines of strong red indicate bad documents, horizontal lines of strong green indicate good documents. When we sort by zELO, we get an almost perfect triangular matrix.

The strong self-consistency of the matrix when sorting by ELO scores indicates the strength of this method. However, this would take O(N^2) Inferences to populate this matrix. Instead, we can employ an optimized sparse sampling strategy that only inferences the ensemble 4 times per document, while still recovering precise zELOs that are within a small error from the zELO induced by the dense matrix.

<img width="677" height="288" alt="image" src="https://github.com/user-attachments/assets/4731f849-e81f-40bd-a211-fd6f273f1f84" />

These zELO scores provide an extremely strong indicator of underlying relevancy, rivaling human annotations in many of our internal ablations, while being orders of magnitude cheaper.

> This method is expected to cost ~$20 / 1000 inferences. And, we inference 4 times per query-document pair. For example, 100 queries with K=25, would cost 100 * 25 * 4 * $20 / 1000 = $200 in calls to OpenAI/Anthropic/Gemini. Note that since each inference involves two documents, each document is involved in ~8 pairwise comparisons.

Additionally, it's highly interpretable. When analyzing the results, you can pick a particular document, and then print out the ~8 pairwise comparisons that involved that document, in order to analyze the ensemble's annotations manually. This can done to understand failure modes of your existing retrieval system. Or, if you disagree with the annotations, it can be used to drive any custom prompt engineering of the Ensemble.

## Setup

First, you'll want to setup the python dependencies.

- Install [Astral UV](https://docs.astral.sh/uv/getting-started/installation/) for virtual environment management.
- Run `uv sync` in order to initialize the virtual environment.
- Run `source .venv/bin/activate` in order to source the virtual environment, which will let you run future python commands.

For annotation purposes, zbench is going to call an ensemble of OpenAI GPT-4, Anthropic Claude, and Google Gemini. To make this work, create a `.env` file in the root directory with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
ZEROENTROPY_API_KEY=your_zeroentropy_api_key # This API Key is optional, but can be used to test ZeroEntropy's retrieval system and reranker models.
```

## Quick Start

The best way to start is by working through [example.ipynb](example.ipynb) using Jupyter or VSCode. The notebook will walk you through the calling zELO annotator, along with benchmarking a retrieval system and graphing NDCG and printing Recall. The individual steps will also be written here:

### 1. Annotate a Dataset

```bash
# Will read from your_dataset.jsonl, and then write the zELO-scored documents to data/your_dataset_zelo_annotated.jsonl
python zbench.annotation data/your_dataset.jsonl data/your_dataset_zelo_annotated.jsonl
```

### 2. Benchmark a Reranker

```python
import asyncio
from zbench.benchmark import benchmark_ndcg, benchmark_accuracy, recall_at_k
from zbench.rerankers import Zerank

async def main():
    # The zELO-annotated documents to use as ground truth
    ZELO_ANNOTATED_DATASET_PATH = "./data/my_sample_dataset_zelo_annotated.jsonl"

    # Use zerank to annotated the dataset
    ZERANK_ANNOTATED_DATASET_PATH = "./data/my_sample_dataset_zerank_annotated.jsonl"
    test_reranker = Zerank("zerank-1")
    await test_reranker.annotate(ZELO_ANNOTATED_DATASET_PATH, ZERANK_ANNOTATED_DATASET_PATH)
    
    # Benchmark with multiple metrics
    ndcg_scores = benchmark_ndcg(ZELO_ANNOTATED_DATASET_PATH, ZERANK_ANNOTATED_DATASET_PATH, visualize=True)
    accuracy_scores = benchmark_accuracy(ZELO_ANNOTATED_DATASET_PATH, ZERANK_ANNOTATED_DATASET_PATH, ground_truth)
    recall_scores = recall_at_k(ZELO_ANNOTATED_DATASET_PATH, ZERANK_ANNOTATED_DATASET_PATH, ground_truth, k=5)
    
    print(f"Average NDCG: {sum(ndcg_scores) / len(ndcg_scores):.4f}")
    print(f"Average Accuracy: {sum(accuracy_scores) / len(accuracy_scores):.4f}")
    print(f"Average Recall@5: {sum(recall_scores) / len(recall_scores):.4f}")

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

The annotation process produces a JSONL file named the same way as an input but is commonly stored with an "_annotated" suffix. Each line contains scored documents in the format of `zbench.common_types.AnnotatedQueryDocuments`:

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
- `--document_threshold`: Maximum number of documents to use per query (default: No Limit). Restricts reranking to the first `--document_threshold` documents in the input order

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

Create a custom reranker by inheriting from `BaseReranker`:

```python
from zbench.common_types import BaseReranker, RerankerInput

class MyCustomReranker(BaseReranker):
    async def score(self, input: RerankerInput) -> list[float]:
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

# Use your custom reranker
my_reranker = MyCustomReranker()
```

Note: Custom rerankers are called simultaneously for every query, so make sure to utilize `asyncio.Semaphore` to avoid ratelimit issues.

### Built-in Rerankers

#### Zerank (ZeroEntropy)
```python
from zbench.rerankers import Zerank

# Available models: "zerank-1", "zerank-1-small"
zerank = Zerank("zerank-1")
zerank_small = Zerank("zerank-1-small")
```

## Evaluation and Benchmarking

zbench provides three evaluation metrics for comprehensive reranker assessment. The first argument is the ground truth, and the second argument is the alternative retrieval system to analayze. The ground truth can be human-annotated binary scores, or it can be Ensemble-annotated zELO scores. Both arguments must be a `AnnotatedQueryDocuments` jsonl file.

### 1. NDCG (Normalized Discounted Cumulative Gain)
Measures ranking quality with position-aware scoring:

```python
# Use with benchmarking functions
ndcg_scores = benchmark_ndcg("dataset_groundtruth.jsonl", "dataset_alternative.jsonl")
```

### 2. Pairwise Accuracy
Measures how often the reranker correctly orders document pairs compared to ground truth:

```python
accuracy_scores = benchmark_accuracy("dataset_groundtruth.jsonl", "dataset_alternative.jsonl")
```

### 3. Recall@K  
Measures what percent of the top-K_gt ground truth documents appear in the reranker's top-K (default value of k_gt = k) results:

```python
recall_scores = recall_at_k("dataset_groundtruth.jsonl", "dataset_alternative.jsonl", k=5, k_gt = 5)
```

### Performance Options

All benchmark functions support a `document_limit` parameter to process only the first N documents per query for faster evaluation:

```python
# Process only first 5 documents per query
ndcg_scores = benchmark_ndcg("dataset.jsonl", test_reranker, ground_truth, document_limit=5)
```

### Visualization

Enable visualization to see NDCG score distributions:

```python
scores = benchmark_reranker(
    "annotated_dataset.jsonl", 
    MY_RERANKER,
    visualize=True,  # Shows histogram of NDCG scores
)
```

### Working with Large Datasets

For large datasets, we recommend:

1. **Increase document threshold gradually**: Start with 10, increase if needed
2. **Use fewer cycles for initial testing**: In practice, no more than 4 cycles are needed for the ELO convergence, but you can lower to 2-3 for small document samples
3. **Monitor API costs**: Each pair requires 3 AI model calls, costing approximately: $20 USD / 1000 pairwise comparisons
4. **Implement checkpointing**: Save intermediate results, do not run on thousands of queries right away. Instead, scale each run by powers of ten and check evaluations and chain of thought at each step to ensure it's working as you intend.
