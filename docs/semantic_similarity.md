# Semantic Similarity Documentation

## Overview

The `SemanticSimilarity` class provides semantic similarity computation using sentence transformers, specifically optimized for query expansion tasks in information retrieval. It computes dense vector representations and cosine similarities between text pairs.

## Quick Start

```python
from src.core.semantic_similarity import SemanticSimilarity

# Initialize with default model
semantic_sim = SemanticSimilarity()

# Compute similarity between two texts
similarity = semantic_sim.compute_similarity("machine learning", "neural networks")
print(f"Similarity: {similarity:.3f}")

# Batch compute similarities for query expansion
query = "machine learning algorithms"
expansion_terms = ["neural", "networks", "classification", "regression"]
similarities = semantic_sim.compute_query_expansion_similarities(query, expansion_terms)
print(similarities)
# Output: {"neural": 0.72, "networks": 0.68, "classification": 0.54, "regression": 0.49}
```

## API Reference

### SemanticSimilarity Class

#### Constructor

```python
SemanticSimilarity(
    model_name: str = 'all-MiniLM-L6-v2',  # Sentence transformer model
    cache_size: int = 1000                  # LRU cache size for embeddings
)
```

**Model Selection Guidelines:**
- `all-MiniLM-L6-v2`: Fast, lightweight, good for development (default)
- `all-mpnet-base-v2`: Higher quality, slower, recommended for production
- `msmarco-distilbert-base-v4`: Optimized for MS MARCO retrieval tasks
- `msmarco-bert-base-dot-v5`: Optimized for dot product similarity

#### Core Methods

**compute_similarity()**
```python
compute_similarity(
    text1: str,                    # First text
    text2: str,                    # Second text  
    metric: str = 'cosine'         # Similarity metric ('cosine', 'dot', 'euclidean')
) -> float                         # Similarity score
```

**compute_query_expansion_similarities()**
```python
compute_query_expansion_similarities(
    query: str,                    # Original query
    expansion_terms: List[str]     # List of expansion terms
) -> Dict[str, float]              # {term: similarity_score}
```

**encode()**
```python
encode(
    texts: Union[str, List[str]],  # Text(s) to encode
    batch_size: int = 32           # Batch size for processing
) -> Union[np.ndarray, List[np.ndarray]]  # Dense embeddings
```

**clear_cache()**
```python
clear_cache() -> None              # Clear embedding cache
```

## Similarity Metrics

### Cosine Similarity (Default)
- **Range**: [-1, 1], typically [0, 1] for sentence embeddings
- **Interpretation**: 1 = identical, 0 = orthogonal, -1 = opposite
- **Use case**: Most common for text similarity

```python
similarity = semantic_sim.compute_similarity("dog", "puppy", metric='cosine')
# Output: ~0.78
```

### Dot Product Similarity  
- **Range**: Unbounded, depends on vector magnitudes
- **Interpretation**: Higher values = more similar
- **Use case**: When vector magnitudes encode importance

```python
similarity = semantic_sim.compute_similarity("dog", "puppy", metric='dot')
# Output: ~0.45 (depends on embedding magnitudes)
```

### Euclidean Distance
- **Range**: [0, ∞), lower is more similar
- **Interpretation**: 0 = identical, higher = more different
- **Use case**: When distance matters more than angle

```python
distance = semantic_sim.compute_similarity("dog", "puppy", metric='euclidean')
# Output: ~0.67 (lower = more similar)
```

## Model Selection Guide

### For Development and Prototyping

```python
# Fast, lightweight model for quick experiments
semantic_sim = SemanticSimilarity('all-MiniLM-L6-v2')
```

**Characteristics:**
- **Speed**: Very fast (384-dim embeddings)
- **Quality**: Good for most tasks
- **Memory**: Low memory usage
- **Use case**: Development, ablation studies

### For Production Quality

```python
# High-quality model for best results
semantic_sim = SemanticSimilarity('all-mpnet-base-v2')
```

**Characteristics:**
- **Speed**: Moderate (768-dim embeddings)
- **Quality**: Excellent semantic understanding
- **Memory**: Moderate memory usage
- **Use case**: Final experiments, production systems

### For IR-Specific Tasks

```python
# Optimized for information retrieval
semantic_sim = SemanticSimilarity('msmarco-distilbert-base-v4')
```

**Characteristics:**
- **Speed**: Fast (768-dim embeddings)
- **Quality**: Optimized for retrieval tasks
- **Training**: Trained on MS MARCO dataset
- **Use case**: MS MARCO experiments, passage retrieval

## Advanced Usage

### Integration with Query Expansion

```python
from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity

# Initialize components
rm_expansion = RMExpansion()
semantic_sim = SemanticSimilarity('all-mpnet-base-v2')

# Complete expansion pipeline
def compute_expansion_importance(query, pseudo_docs, scores):
    # Get RM expansion terms
    rm_terms = rm_expansion.expand_query(query, pseudo_docs, scores)
    
    # Compute semantic similarities
    expansion_words = [term for term, weight in rm_terms]
    semantic_scores = semantic_sim.compute_query_expansion_similarities(query, expansion_words)
    
    # Combine RM weights with semantic scores
    importance_weights = {}
    for term, rm_weight in rm_terms:
        semantic_score = semantic_scores.get(term, 0.0)
        # Simple combination (α=1.0, γ=1.0)
        importance = rm_weight + semantic_score
        importance_weights[term] = importance
    
    return importance_weights
```

### Batch Processing for Efficiency

```python
# Process multiple queries efficiently
queries = ["machine learning", "neural networks", "deep learning"]
all_similarities = {}

# Batch encode all queries once
query_embeddings = semantic_sim.encode(queries)

# Compute similarities for each query
expansion_terms = ["algorithm", "model", "training", "data"]
for i, query in enumerate(queries):
    similarities = {}
    term_embeddings = semantic_sim.encode(expansion_terms)
    
    # Compute cosine similarities
    from sentence_transformers.util import cos_sim
    scores = cos_sim(query_embeddings[i:i+1], term_embeddings)[0]
    
    for j, term in enumerate(expansion_terms):
        similarities[term] = float(scores[j])
    
    all_similarities[query] = similarities
```

### Custom Similarity Functions

```python
class CustomSemanticSimilarity(SemanticSimilarity):
    def compute_weighted_similarity(self, query, term, query_weight=1.0, term_weight=1.0):
        """Compute similarity with custom weighting."""
        # Get embeddings
        query_emb = self.encode([query])[0]
        term_emb = self.encode([term])[0]
        
        # Apply weights
        weighted_query = query_emb * query_weight
        weighted_term = term_emb * term_weight
        
        # Compute cosine similarity
        from sentence_transformers.util import cos_sim
        return float(cos_sim(weighted_query, weighted_term))
    
    def compute_contextual_similarity(self, query, term, context_docs):
        """Compute similarity considering document context."""
        # Create contextualized representations
        query_context = f"{query} {' '.join(context_docs[:2])}"  # Add context
        
        return self.compute_similarity(query_context, term)
```

## Performance Optimization

### Caching Strategy

```python
# Large cache for repeated term lookups
semantic_sim = SemanticSimilarity(cache_size=10000)

# Check cache performance
if hasattr(semantic_sim._encode_single, 'cache_info'):
    info = semantic_sim._encode_single.cache_info()
    print(f"Cache hits: {info.hits}/{info.hits + info.misses} ({info.hits/(info.hits + info.misses)*100:.1f}%)")
```

### Memory Management

```python
# Clear cache periodically for long-running processes
semantic_sim = SemanticSimilarity()

for batch_num, queries in enumerate(query_batches):
    # Process batch
    results = process_batch(queries)
    
    # Clear cache every 100 batches
    if batch_num % 100 == 0:
        semantic_sim.clear_cache()
        print(f"Cleared cache after batch {batch_num}")
```

### GPU Optimization

```python
# For large-scale processing, use GPU
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    semantic_sim = SemanticSimilarity('all-mpnet-base-v2')
else:
    print("Using CPU")
    semantic_sim = SemanticSimilarity('all-MiniLM-L6-v2')  # Faster on CPU
```

## Integration Examples

### With Weight Optimization

```python
from src.models.weight_optimizer import LBFGSOptimizer

def create_evaluation_function(semantic_sim, rm_expansion, queries, qrels):
    def evaluate_weights(weights):
        alpha, beta, gamma = weights
        total_score = 0.0
        
        for query_id, query_text in queries.items():
            # Get expansion terms
            expansion_terms = rm_expansion.expand_query(query_text, pseudo_docs, scores)
            
            # Compute semantic similarities
            expansion_words = [term for term, weight in expansion_terms]
            semantic_scores = semantic_sim.compute_query_expansion_similarities(
                query_text, expansion_words
            )
            
            # Apply learned weights
            for term, rm_weight in expansion_terms:
                semantic_score = semantic_scores.get(term, 0.0)
                importance = alpha * rm_weight + gamma * semantic_score
                # Use importance for retrieval...
            
            # Evaluate retrieval performance
            score = evaluate_retrieval_performance(query_id, importance_weights, qrels)
            total_score += score
        
        return total_score / len(queries)
    
    return evaluate_weights

# Use in optimization
optimizer = LBFGSOptimizer()
eval_func = create_evaluation_function(semantic_sim, rm_expansion, val_queries, val_qrels)
optimal_weights = optimizer.optimize_weights(training_data, val_queries, val_qrels, eval_func)
```

### With Multi-Vector Retrieval

```python
from src.models.multivector_retrieval import MultiVectorRetrieval

# Initialize multi-vector retrieval with semantic similarity
retrieval_system = MultiVectorRetrieval('all-MiniLM-L6-v2')
semantic_sim = SemanticSimilarity('all-MiniLM-L6-v2')  # Same model

def create_importance_weighted_query(query, expansion_terms, importance_weights):
    query_vectors = []
    
    # Original query tokens
    query_tokens = retrieval_system.tokenizer.tokenize(query)
    for token in query_tokens:
        embedding = retrieval_system._get_token_embedding(token)
        query_vectors.append(1.0 * embedding)  # Baseline importance
    
    # Expansion term vectors with importance weighting
    for term, rm_weight in expansion_terms:
        importance = importance_weights.get(term, 0.0)
        
        # Tokenize expansion term
        term_tokens = retrieval_system.tokenizer.tokenize(term)
        for token in term_tokens:
            embedding = retrieval_system._get_token_embedding(token)
            scaled_embedding = importance * embedding  # Scale by importance
            query_vectors.append(scaled_embedding)
    
    return query_vectors
```

## Quality Assessment

### Evaluation Metrics

```python
def evaluate_semantic_similarity_quality(semantic_sim, test_pairs):
    """Evaluate semantic similarity quality on test pairs."""
    similarities = []
    human_ratings = []
    
    for text1, text2, human_rating in test_pairs:
        sim_score = semantic_sim.compute_similarity(text1, text2)
        similarities.append(sim_score)
        human_ratings.append(human_rating)
    
    # Compute correlation with human judgments
    import numpy as np
    correlation = np.corrcoef(similarities, human_ratings)[0, 1]
    
    return {
        'correlation': correlation,
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities)
    }

# Example test pairs (text1, text2, human_rating)
test_pairs = [
    ("dog", "puppy", 0.8),
    ("car", "automobile", 0.9),
    ("happy", "joyful", 0.85),
    ("computer", "banana", 0.1)
]

quality_metrics = evaluate_semantic_similarity_quality(semantic_sim, test_pairs)
print(f"Correlation with human ratings: {quality_metrics['correlation']:.3f}")
```

### Query Expansion Quality

```python
def analyze_expansion_quality(semantic_sim, queries, ground_truth_terms):
    """Analyze quality of semantic similarity for query expansion."""
    results = {}
    
    for query, relevant_terms in ground_truth_terms.items():
        # Compute similarities
        similarities = semantic_sim.compute_query_expansion_similarities(query, relevant_terms)
        
        # Analyze distribution
        scores = list(similarities.values())
        results[query] = {
            'mean_similarity': np.mean(scores),
            'std_similarity': np.std(scores),
            'min_similarity': min(scores),
            'max_similarity': max(scores),
            'high_quality_terms': [term for term, score in similarities.items() if score > 0.6]
        }
    
    return results

# Example usage
ground_truth = {
    "machine learning": ["algorithm", "neural", "training", "model", "classification"],
    "information retrieval": ["search", "index", "query", "document", "ranking"]
}

quality_analysis = analyze_expansion_quality(semantic_sim, ground_truth.keys(), ground_truth)
for query, metrics in quality_analysis.items():
    print(f"{query}: {len(metrics['high_quality_terms'])} high-quality terms")
```

## Troubleshooting

### Common Issues

**1. Model Download Fails**
```python
# Manual model download
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models/')

# Or use local model
semantic_sim = SemanticSimilarity('./path/to/local/model')
```

**2. Out of Memory Errors**
```python
# Use smaller batch sizes
semantic_sim = SemanticSimilarity('all-MiniLM-L6-v2')  # Smaller model
similarities = semantic_sim.encode(texts, batch_size=8)  # Smaller batches

# Clear cache frequently
semantic_sim.clear_cache()
```

**3. Slow Performance**
```python
# Use faster model
semantic_sim = SemanticSimilarity('all-MiniLM-L6-v2')  # Instead of all-mpnet-base-v2

# Increase cache size for repeated computations
semantic_sim = SemanticSimilarity(cache_size=5000)

# Use GPU if available
import torch
if torch.cuda.is_available():
    # Model automatically uses GPU
    semantic_sim = SemanticSimilarity('all-MiniLM-L6-v2')
```

**4. Unexpected Similarity Scores**
```python
# Debug similarity computation
query = "machine learning"
term = "neural networks"

# Check embeddings
query_emb = semantic_sim.encode([query])[0]
term_emb = semantic_sim.encode([term])[0]

print(f"Query embedding shape: {query_emb.shape}")
print(f"Term embedding shape: {term_emb.shape}")
print(f"Query embedding norm: {np.linalg.norm(query_emb):.3f}")
print(f"Term embedding norm: {np.linalg.norm(term_emb):.3f}")

# Compute similarity manually
from sentence_transformers.util import cos_sim
manual_sim = cos_sim(query_emb, term_emb).item()
api_sim = semantic_sim.compute_similarity(query, term)

print(f"Manual similarity: {manual_sim:.3f}")
print(f"API similarity: {api_sim:.3f}")
assert abs(manual_sim - api_sim) < 1e-6, "Similarity computation mismatch"
```

## Testing

Run semantic similarity tests:

```bash
# Run all semantic similarity tests
python -m pytest tests/test_semantic_similarity.py -v

# Test specific functionality
python -c "
from src.core.semantic_similarity import SemanticSimilarity
sem_sim = SemanticSimilarity()
score = sem_sim.compute_similarity('dog', 'puppy')
print(f'Similarity: {score:.3f}')
assert 0.5 < score < 1.0, f'Unexpected similarity: {score}'
print('Semantic similarity test passed!')
"
```

## Best Practices

### Model Selection
1. **Development**: Start with `all-MiniLM-L6-v2` for speed
2. **Production**: Use `all-mpnet-base-v2` for quality
3. **IR Tasks**: Consider `msmarco-distilbert-base-v4` for MS MARCO
4. **Custom Domains**: Fine-tune on domain-specific data

### Performance Optimization
1. **Cache Size**: Set based on vocabulary size (1000-10000)
2. **Batch Processing**: Use batch encoding for multiple texts
3. **GPU Usage**: Enable for large-scale processing
4. **Memory Management**: Clear cache periodically

### Quality Assurance
1. **Validate on Test Sets**: Use human-annotated similarity pairs
2. **Domain Adaptation**: Test on domain-specific query-term pairs
3. **Correlation Analysis**: Check correlation with retrieval performance
4. **Ablation Studies**: Compare different models and metrics

This comprehensive guide should help you effectively use semantic similarity in your expansion weight learning system!