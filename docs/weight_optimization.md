# Weight Optimization Documentation

## Overview

The weight optimization module provides algorithms for learning optimal importance weights (α, β, γ) that combine three complementary signals in query expansion: RM weights, BM25 scores, and semantic similarity scores.

## Core Methodology

### Problem Formulation

We optimize the importance scoring function:

```
importance(term) = α × RM_weight + β × BM25_score + γ × semantic_similarity
```

**Objective**: Find optimal weights that maximize retrieval performance on validation data.

### Key Insight

In multi-vector retrieval, **vector magnitude encodes importance**. By scaling expansion term embeddings by their importance scores, we naturally emphasize query-relevant terms in the late interaction process.

## Quick Start

```python
from src.models.weight_optimizer import LBFGSOptimizer, create_optimizer

# Initialize optimizer
optimizer = LBFGSOptimizer()

# Create evaluation function (your retrieval pipeline)
def evaluate_weights(weights):
    alpha, beta, gamma = weights
    # Run your retrieval evaluation with these weights
    return retrieval_performance_score

# Learn optimal weights
optimal_weights = optimizer.optimize_weights(
    training_data=training_data,
    validation_queries=validation_queries,  
    validation_qrels=validation_qrels,
    evaluation_function=evaluate_weights
)

print(f"Learned weights: α={optimal_weights[0]:.3f}, β={optimal_weights[1]:.3f}, γ={optimal_weights[2]:.3f}")
```

## Available Optimizers

### 1. LBFGSOptimizer (Recommended)

**Best for**: Production use, fast convergence, smooth evaluation functions

```python
from src.models.weight_optimizer import LBFGSOptimizer

optimizer = LBFGSOptimizer(
    bounds=[(0.1, 5.0)] * 3,    # Weight bounds for (α, β, γ)
    max_iterations=50           # Maximum optimization iterations
)
```

**Characteristics:**
- **Algorithm**: Limited-memory BFGS with box constraints
- **Convergence**: Fast (typically 20-50 function evaluations)
- **Quality**: High-quality local optima
- **Use case**: Production systems, smooth evaluation functions

### 2. GridSearchOptimizer

**Best for**: Comprehensive search, guaranteed global optimum, baseline comparisons

```python
from src.models.weight_optimizer import GridSearchOptimizer

optimizer = GridSearchOptimizer(
    weight_ranges=[[0.1, 3.0]] * 3,  # Search ranges for each weight
    resolution=10                     # Grid resolution (10³ = 1000 evaluations)
)
```

**Characteristics:**
- **Algorithm**: Exhaustive grid search
- **Evaluations**: Fixed number (resolution³)
- **Quality**: Guaranteed global optimum within grid resolution
- **Use case**: Baseline comparisons, noisy evaluation functions

### 3. RandomSearchOptimizer

**Best for**: Quick baselines, noisy functions, exploration

```python
from src.models.weight_optimizer import RandomSearchOptimizer

optimizer = RandomSearchOptimizer(
    num_samples=1000,                # Number of random samples
    weight_ranges=[[0.1, 3.0]] * 3  # Sampling ranges
)
```

**Characteristics:**
- **Algorithm**: Random sampling from uniform distribution
- **Evaluations**: Fixed number (num_samples)
- **Quality**: Good baseline, probabilistic convergence
- **Use case**: Quick exploration, noisy evaluation functions

## API Reference

### Base WeightOptimizer Class

```python
from src.models.weight_optimizer import WeightOptimizer

class WeightOptimizer(ABC):
    @abstractmethod
    def optimize_weights(
        self,
        training_data: Dict[str, Any],              # Training dataset
        validation_queries: Dict[str, str],         # {query_id: query_text}
        validation_qrels: Dict[str, Dict[str, int]], # {query_id: {doc_id: relevance}}
        evaluation_function: Callable               # Function that takes weights and returns performance
    ) -> Tuple[float, float, float]:               # Returns (α, β, γ)
```

### LBFGSOptimizer

```python
LBFGSOptimizer(
    bounds: List[Tuple[float, float]] = [(0.1, 5.0)] * 3,  # Box constraints
    max_iterations: int = 50                                # Maximum iterations
)

# Methods
optimizer.optimize_weights(training_data, queries, qrels, eval_func)
```

### GridSearchOptimizer

```python
GridSearchOptimizer(
    weight_ranges: List[List[float]] = [[0.1, 3.0]] * 3,  # Search ranges
    resolution: int = 10                                    # Grid resolution
)

# Attributes after optimization
optimizer.evaluation_history  # List of (weights, score) tuples
optimizer.best_weights        # Best weights found
optimizer.best_score         # Best score achieved
```

### RandomSearchOptimizer

```python
RandomSearchOptimizer(
    num_samples: int = 100,                               # Number of samples
    weight_ranges: List[List[float]] = [[0.1, 3.0]] * 3  # Sampling ranges
)

# Attributes after optimization
optimizer.evaluation_history  # List of (weights, score) tuples
```

### Factory Function

```python
from src.models.weight_optimizer import create_optimizer

# Create optimizer by name
optimizer = create_optimizer('lbfgs')                    # LBFGSOptimizer
optimizer = create_optimizer('grid', resolution=8)       # GridSearchOptimizer  
optimizer = create_optimizer('random', num_samples=500)  # RandomSearchOptimizer
```

## Usage Examples

### Basic Usage

```python
from src.models.weight_optimizer import LBFGSOptimizer

# Your evaluation function
def evaluate_weights(weights):
    alpha, beta, gamma = weights
    
    # Apply weights in your retrieval pipeline
    total_score = 0.0
    for query_id, query_text in validation_queries.items():
        # Compute importance-weighted expansion
        importance_weights = {}
        for term, rm_weight in expansion_terms[query_id]:
            bm25_score = get_bm25_score(term, query_text)
            semantic_score = get_semantic_similarity(term, query_text)
            importance = alpha * rm_weight + beta * bm25_score + gamma * semantic_score
            importance_weights[term] = importance
        
        # Run retrieval and evaluate
        results = retrieve_with_importance_weights(query_text, importance_weights)
        score = compute_ndcg(results, validation_qrels[query_id])
        total_score += score
    
    return total_score / len(validation_queries)

# Optimize
optimizer = LBFGSOptimizer()
optimal_weights = optimizer.optimize_weights(
    training_data, validation_queries, validation_qrels, evaluate_weights
)
```

### Comparing Multiple Optimizers

```python
from src.models.weight_optimizer import LBFGSOptimizer, GridSearchOptimizer, RandomSearchOptimizer

optimizers = {
    'L-BFGS-B': LBFGSOptimizer(bounds=[(0.1, 5.0)] * 3),
    'Grid Search': GridSearchOptimizer(weight_ranges=[[0.1, 3.0]] * 3, resolution=8),
    'Random Search': RandomSearchOptimizer(num_samples=500)
}

results = {}
for name, optimizer in optimizers.items():
    optimal_weights = optimizer.optimize_weights(
        training_data, validation_queries, validation_qrels, evaluation_function
    )
    
    # Evaluate on test set
    test_score = evaluate_on_test_set(optimal_weights)
    
    results[name] = {
        'weights': optimal_weights,
        'test_score': test_score
    }
    
    print(f"{name}: α={optimal_weights[0]:.3f}, β={optimal_weights[1]:.3f}, γ={optimal_weights[2]:.3f}")
    print(f"Test score: {test_score:.4f}")
```

### Integration with Existing Components

```python
from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.models.weight_optimizer import LBFGSOptimizer

# Initialize components
rm_expansion = RMExpansion()
semantic_sim = SemanticSimilarity('all-MiniLM-L6-v2')
optimizer = LBFGSOptimizer()

def create_evaluation_function(rm_expansion, semantic_sim, validation_data):
    """Create evaluation function using existing components."""
    
    def evaluate_weights(weights):
        alpha, beta, gamma = weights
        total_performance = 0.0
        
        for query_id, query_text in validation_data['queries'].items():
            # RM expansion
            pseudo_docs = get_pseudo_relevant_docs(query_id, validation_data)
            expansion_terms = rm_expansion.expand_query(query_text, pseudo_docs, [1.0] * len(pseudo_docs))
            
            # Compute importance weights
            expansion_words = [term for term, weight in expansion_terms]
            semantic_scores = semantic_sim.compute_query_expansion_similarities(query_text, expansion_words)
            
            importance_weights = {}
            for term, rm_weight in expansion_terms:
                bm25_score = get_bm25_score(term, query_text)  # Your BM25 implementation
                semantic_score = semantic_scores.get(term, 0.0)
                
                importance = alpha * rm_weight + beta * bm25_score + gamma * semantic_score
                importance_weights[term] = importance
            
            # Retrieval and evaluation
            results = your_retrieval_system(query_text, importance_weights)
            performance = compute_metric(results, validation_data['qrels'][query_id])
            total_performance += performance
        
        return total_performance / len(validation_data['queries'])
    
    return evaluate_weights

# Use the evaluation function
eval_func = create_evaluation_function(rm_expansion, semantic_sim, validation_data)
optimal_weights = optimizer.optimize_weights(training_data, validation_queries, validation_qrels, eval_func)
```

## Algorithm Selection Guide

### Choose L-BFGS-B when:
- Evaluation function is smooth and differentiable
- Need fast convergence (< 50 evaluations)
- Production deployment
- Have reasonable starting point

### Choose Grid Search when:
- Need guaranteed global optimum
- Evaluation function is noisy or discontinuous
- Want comprehensive baseline comparison
- Have computational resources for exhaustive search

### Choose Random Search when:
- Quick baseline needed
- Evaluation function is very noisy
- Initial exploration of search space
- Comparison with sophisticated methods

## Configuration Guidelines

### Bounds Selection

```python
# Conservative bounds (good starting point)
bounds = [(0.1, 3.0)] * 3

# Wide bounds (if hitting boundaries)
bounds = [(0.05, 5.0)] * 3

# Custom bounds per weight
bounds = [(0.1, 2.0), (0.1, 3.0), (0.1, 4.0)]  # Different ranges for α, β, γ
```

### Iteration Limits

```python
# Development (fast iterations)
optimizer = LBFGSOptimizer(max_iterations=25)

# Production (thorough optimization)
optimizer = LBFGSOptimizer(max_iterations=100)
```

### Grid Resolution

```python
# Quick exploration
optimizer = GridSearchOptimizer(resolution=5)    # 5³ = 125 evaluations

# Thorough search  
optimizer = GridSearchOptimizer(resolution=10)   # 10³ = 1,000 evaluations

# Fine-grained search
optimizer = GridSearchOptimizer(resolution=15)   # 15³ = 3,375 evaluations
```

## Best Practices

### 1. Evaluation Function Design

**Guidelines:**
- **Smooth**: Avoid discontinuities that confuse gradient-based optimizers
- **Stable**: Multiple evaluations with same weights should give similar scores
- **Representative**: Use validation set that matches test conditions
- **Efficient**: Minimize evaluation time

### 2. Validation Strategy

```python
# Hold-out validation
train_queries, val_queries = split_queries(all_queries, test_size=0.2)

# Learn weights on train_queries, validate on val_queries
# Test on separate held-out test set (e.g., TREC DL)
```

### 3. Multiple Random Restarts

```python
# For L-BFGS-B: try multiple starting points
best_weights = None
best_score = -float('inf')

for trial in range(5):
    # Random starting point
    start_weights = np.random.uniform(0.5, 2.0, 3)
    
    optimizer = LBFGSOptimizer()
    weights = optimizer.optimize_weights(training_data, queries, qrels, eval_func)
    score = eval_func(weights)
    
    if score > best_score:
        best_score = score
        best_weights = weights
```

### 4. Reproducibility

```python
import numpy as np

# Set random seed for reproducible results
np.random.seed(42)

# For RandomSearchOptimizer
optimizer = RandomSearchOptimizer(num_samples=1000, weight_ranges=[[0.1, 3.0]] * 3)
```

## Troubleshooting

### Common Issues

**1. L-BFGS-B doesn't converge**
```python
# Increase iteration limit
optimizer = LBFGSOptimizer(max_iterations=100)

# Try looser bounds
optimizer = LBFGSOptimizer(bounds=[(0.01, 10.0)] * 3)

# Multiple random starts (see Best Practices above)
```

**2. Evaluation function is noisy**
```python
# Use Random Search (more robust to noise)
optimizer = RandomSearchOptimizer(num_samples=1000)

# Or Grid Search
optimizer = GridSearchOptimizer(resolution=8)
```

**3. Optimization is too slow**
```python
# Use smaller validation set
small_queries = dict(list(validation_queries.items())[:100])

# Use Random Search with fewer samples
optimizer = RandomSearchOptimizer(num_samples=200)

# Reduce Grid Search resolution
optimizer = GridSearchOptimizer(resolution=6)  # 6³ = 216 evaluations
```

**4. Weights hit boundaries**
```python
# Check if optimal weights are at boundaries
optimal_weights = optimizer.optimize_weights(...)
bounds = [(0.1, 5.0)] * 3

for i, (weight, (low, high)) in enumerate(zip(optimal_weights, bounds)):
    if abs(weight - low) < 0.01:
        print(f"Weight {i} hit lower bound: {weight:.3f}")
    elif abs(weight - high) < 0.01:
        print(f"Weight {i} hit upper bound: {weight:.3f}")

# Expand bounds if needed
expanded_bounds = [(0.05, 10.0)] * 3
```

## Testing

Run weight optimizer tests:

```bash
# Run all weight optimizer tests
python -m pytest tests/test_weight_optimizer.py -v

# Test specific optimizer
python tests/test_weight_optimizer.py --optimizer lbfgs

# Quick test
python -c "
from src.models.weight_optimizer import LBFGSOptimizer
def mock_eval(weights): 
    return -sum((w-1)**2 for w in weights)  # Optimum at (1,1,1)
optimizer = LBFGSOptimizer()
result = optimizer.optimize_weights({}, {}, {}, mock_eval)
print(f'Converged to: {result}')
print('Weight optimization test passed!')
"
```

## Integration with Scripts

The weight optimization module is used by the `train_weights.py` script:

```bash
# Use L-BFGS-B optimizer
python scripts/train_weights.py --optimizer lbfgs --training_data ./data --output_dir ./models

# Use Grid Search optimizer  
python scripts/train_weights.py --optimizer grid --training_data ./data --output_dir ./models

# Use Random Search optimizer
python scripts/train_weights.py --optimizer random --training_data ./data --output_dir ./models
```

For detailed script usage, see [`docs/train_weights.md`](train_weights.md).

## Summary

The weight optimization module provides three complementary algorithms for learning optimal importance weights:

- **L-BFGS-B**: Fast, high-quality optimization for production use
- **Grid Search**: Exhaustive search with guaranteed global optimum  
- **Random Search**: Robust baseline for noisy evaluation functions

Choose the algorithm based on your evaluation function characteristics, computational budget, and quality requirements. All optimizers integrate seamlessly with the existing RM expansion, BM25, and semantic similarity components.