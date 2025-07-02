# Weight Training Script Documentation

## Overview

The `train_weights.py` script learns optimal importance weights (α, β, γ) for combining RM expansion, BM25 scores, and semantic similarity in query expansion. It uses TREC DL validation data to optimize retrieval performance and saves the learned weights for production use.

## Quick Start

```bash
# Basic usage (semantic similarity only)
python scripts/train_weights.py \
    --training_data ./training_data \
    --output_dir ./models

# With BM25 integration 
python scripts/train_weights.py \
    --training_data ./training_data \
    --index_path ./indexes/msmarco-passage_bert-base-uncased \
    --lucene_path /path/to/lucene/* \
    --output_dir ./models
```

## Command Line Arguments

Based on the actual `train_weights.py` script:

### Required Arguments

- `--training_data`: Path to training data directory (created by `create_training_data.py`)
- `--output_dir`: Output directory for learned weights and results

### Data Parameters

- `--validation_dataset`: Validation dataset name (default: `msmarco-passage/trec-dl-2019`)
  - Options: `msmarco-passage/trec-dl-2019`, `msmarco-passage/trec-dl-2020`

### BM25 Parameters

- `--index_path`: Path to BM25 index (optional)
- `--lucene_path`: Path to Lucene JAR files (required if using `--index_path`)

### Model Parameters

- `--semantic_model`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `--optimizer`: Optimization algorithm (default: `lbfgs`)
  - Options: `lbfgs`, `grid`, `random`
- `--metric`: Metric to optimize (default: `ndcg_cut_10`)

### Logging Parameters

- `--log_level`: Logging verbosity (default: `INFO`)

## Core Components

The script uses these existing classes from your codebase:

1. **WeightTrainer** class with these methods:
   - `load_validation_data()`: Loads TREC DL data via ir_datasets
   - `compute_expansion_terms_dict()`: Computes RM expansion for all queries
   - `create_evaluation_function()`: Creates the optimization objective
   - `train_weights()`: Main training loop

2. **Component Integration**:
   - `RMExpansion` from `src.core.rm_expansion`
   - `SemanticSimilarity` from `src.core.semantic_similarity`
   - `TokenBM25Scorer` from `bert_bm25_scorer` (optional)
   - Weight optimizers from `src.models.weight_optimizer`
   - `TRECDLReranker` from `src.models.multivector_reranking`

## Training Workflow

### 1. Data Loading

```python
# Load training data created by create_training_data.py
training_data = load_training_data(args.training_data)

# Load validation data from TREC DL via ir_datasets
validation_data = trainer.load_validation_data(args.validation_dataset)
```

### 2. Component Initialization

```python
# Initialize components
bm25_scorer = TokenBM25Scorer(args.index_path) if args.index_path else None
semantic_sim = SemanticSimilarity(args.semantic_model)
rm_expansion = RMExpansion()
reranker = TRECDLReranker(args.semantic_model)

# Create trainer
trainer = WeightTrainer(bm25_scorer, semantic_sim, rm_expansion, reranker)
```

### 3. Weight Optimization Process

The actual implementation follows this process:

```python
def train_weights(self, training_data, validation_data, optimizer_type, metric):
    # 1. Compute RM expansion terms for all validation queries
    expansion_terms_dict = self.compute_expansion_terms_dict(
        validation_data['queries'],
        validation_data['first_stage_runs'], 
        validation_data['documents']
    )
    
    # 2. Create evaluation function
    evaluation_function = self.create_evaluation_function(
        validation_data, expansion_terms_dict, metric
    )
    
    # 3. Test baseline performance
    baseline_weights = (1.0, 1.0, 1.0)
    baseline_score = evaluation_function(baseline_weights)
    
    # 4. Optimize weights
    optimizer = create_optimizer(optimizer_type)
    optimal_weights = optimizer.optimize_weights(
        training_data, 
        validation_data['queries'],
        validation_data['qrels'],
        evaluation_function
    )
    
    return optimal_weights
```

### 4. Evaluation Function

The key evaluation function in the actual code:

```python
def evaluate_weights(weights):
    alpha, beta, gamma = weights
    
    # Compute importance weights for all queries
    importance_weights_dict = {}
    for query_id, query_text in queries.items():
        expansion_terms = expansion_terms_dict[query_id]
        
        # Get semantic similarities (batch computation)
        expansion_words = [term for term, weight in expansion_terms]
        semantic_scores = self.semantic_sim.compute_query_expansion_similarities(
            query_text, expansion_words
        )
        
        # Compute importance for each term
        importance_weights = {}
        for term, rm_weight in expansion_terms:
            # BM25 score (if available)
            bm25_score = 0.0
            if self.bm25_scorer and reference_doc_id:
                bm25_scores = self.bm25_scorer.compute_bm25_term_weight(reference_doc_id, [term])
                bm25_score = float(bm25_scores.get(term, 0.0))
            
            # Semantic score
            semantic_score = semantic_scores.get(term, 0.0)
            
            # Combine using current weights
            importance = alpha * rm_weight + beta * bm25_score + gamma * semantic_score
            importance_weights[term] = importance
        
        importance_weights_dict[query_id] = importance_weights
    
    # Rerank using multi-vector method
    reranked_results = self.reranker.rerank_trec_dl_run(
        queries=queries,
        first_stage_runs=first_stage_runs,
        expansion_terms_dict=expansion_terms_dict,
        importance_weights_dict=importance_weights_dict,
        top_k=100
    )
    
    # Evaluate performance
    evaluator = create_trec_dl_evaluator()
    evaluation = evaluator.evaluate_run(reranked_results, qrels)
    
    return evaluation.get(metric, 0.0)
```

## Output Structure

The script creates this output structure:

```
output_dir/
├── learned_weights.json      # Main output: optimized weights
├── weight_training_YYYYMMDD_HHMMSS/
│   ├── results.json          # Complete experiment results
│   └── summary.json          # Experiment summary
```

### Learned Weights Format

From the actual `save_learned_weights()` implementation:

```json
{
  "weights": {
    "alpha": 1.247,
    "beta": 0.823, 
    "gamma": 1.156
  },
  "created_at": "2025-01-15T14:30:45.123456",
  "experiment_info": {
    "training_data": "./training_data",
    "validation_dataset": "msmarco-passage/trec-dl-2019",
    "semantic_model": "all-mpnet-base-v2",
    "optimizer": "lbfgs",
    "metric": "ndcg_cut_10",
    "bm25_available": true
  }
}
```

## Usage Examples

### Basic Training

```bash
# Train with default settings
python scripts/train_weights.py \
    --training_data ./training_data \
    --output_dir ./models

# Train with specific semantic model
python scripts/train_weights.py \
    --training_data ./training_data \
    --semantic_model all-mpnet-base-v2 \
    --output_dir ./models
```

### Training with BM25 Integration

```bash
# Full pipeline with BM25
python scripts/train_weights.py \
    --training_data ./training_data \
    --index_path ./indexes/msmarco-passage_bert-base-uncased \
    --lucene_path /path/to/lucene/* \
    --semantic_model all-mpnet-base-v2 \
    --optimizer lbfgs \
    --output_dir ./models
```

### Different Optimizers

```bash
# L-BFGS-B optimizer (default, recommended)
python scripts/train_weights.py \
    --training_data ./training_data \
    --optimizer lbfgs \
    --output_dir ./models

# Grid search optimizer  
python scripts/train_weights.py \
    --training_data ./training_data \
    --optimizer grid \
    --output_dir ./models

# Random search optimizer
python scripts/train_weights.py \
    --training_data ./training_data \
    --optimizer random \
    --output_dir ./models
```

### Different Validation Datasets

```bash
# Train on TREC DL 2019 (default)
python scripts/train_weights.py \
    --training_data ./training_data \
    --validation_dataset msmarco-passage/trec-dl-2019 \
    --output_dir ./models

# Train on TREC DL 2020
python scripts/train_weights.py \
    --training_data ./training_data \
    --validation_dataset msmarco-passage/trec-dl-2020 \
    --output_dir ./models
```

## Integration with Other Scripts

### Complete Pipeline

```bash
# 1. Create training data
python scripts/create_training_data.py \
    --output_dir ./training_data \
    --max_queries 10000

# 2. Train weights
python scripts/train_weights.py \
    --training_data ./training_data \
    --output_dir ./models

# 3. Use learned weights (in your evaluation code)
from src.utils.file_utils import load_learned_weights
weights = load_learned_weights('./models/learned_weights.json')
alpha, beta, gamma = weights
```

### Loading and Using Weights

```python
# Load learned weights
from src.utils.file_utils import load_learned_weights

weights = load_learned_weights('./models/learned_weights.json')
alpha, beta, gamma = weights

print(f"Learned weights: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")

# Use in expansion model
from src.models.expansion_models import ImportanceWeightedExpansionModel

expansion_model = ImportanceWeightedExpansionModel(
    rm_expansion=rm_expansion,
    semantic_similarity=semantic_sim,
    bm25_scorer=bm25_scorer,
    alpha=alpha,
    beta=beta, 
    gamma=gamma
)
```

## Error Handling and Troubleshooting

### Common Issues

**1. Training Data Not Found**
```bash
# Error: FileNotFoundError
# Solution: Create training data first
python scripts/create_training_data.py --output_dir ./training_data
```

**2. Lucene Initialization Failed**
```bash
# Error: Failed to initialize Lucene JVM
# Solution: Check Lucene setup
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
python scripts/train_weights.py \
    --training_data ./training_data \
    --lucene_path /correct/path/to/lucene/* \
    --output_dir ./models
```

**3. BM25 Scorer Not Available**
```bash
# Warning: BM25 scorer not available despite successful Lucene initialization
# This means the bert_bm25_scorer module couldn't be imported
# Check if the BM25 indexer is properly set up
```

**4. No Validation Queries Found**
```bash
# Error: No validation queries found  
# Solution: Check dataset name
python scripts/train_weights.py \
    --training_data ./training_data \
    --validation_dataset msmarco-passage/trec-dl-2019 \
    --output_dir ./models
```

### Debug Mode

```bash
# Enable debug logging
python scripts/train_weights.py \
    --training_data ./training_data \
    --output_dir ./models \
    --log_level DEBUG
```

## Expected Output

When running successfully, you should see:

```
INFO - WeightTrainer initialized
INFO - BM25 available: True
INFO - Loading validation data: msmarco-passage/trec-dl-2019
INFO - Loaded 43 validation queries
INFO - Loaded qrels for 43 queries  
INFO - Loaded first-stage runs for 43 queries
INFO - Loaded 8,841,823 documents
INFO - Computing RM expansion terms for validation queries...
INFO - Computed expansion terms for 43 queries
INFO - Evaluating baseline performance...
INFO - Baseline performance (ndcg_cut_10): 0.5064
INFO - Starting weight optimization...
INFO - L-BFGS-B optimization completed!
INFO - Optimal weights: alpha=1.247, beta=0.823, gamma=1.156
INFO - Optimal score: 0.5423
INFO - Improvement: +0.0359
INFO - Weight training completed successfully!
INFO - Results saved to: ./models
```

## Performance Considerations

### Memory Usage

The script loads the entire document collection into memory. For MS MARCO:
- ~8.8M documents ≈ 12-16GB RAM
- Use machines with at least 32GB RAM for comfortable operation

### Training Time

Typical training times:
- L-BFGS-B: 10-30 minutes (recommended)
- Grid Search: 2-4 hours (exhaustive)
- Random Search: 30-60 minutes

### GPU Usage

The semantic similarity component automatically uses GPU if available:
```bash
# Check GPU usage
nvidia-smi --loop=5

# Force CPU usage if needed
export CUDA_VISIBLE_DEVICES=""
python scripts/train_weights.py --training_data ./training_data --output_dir ./models
```
