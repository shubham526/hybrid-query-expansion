# RM Expansion Documentation

## Overview

The `RMExpansion` class implements Relevance Model (RM) query expansion algorithms, including RM1 and RM3 variants. These are fundamental pseudo-relevance feedback techniques that expand queries using terms from top-retrieved documents.

## Quick Start

```python
from src.core.rm_expansion import RMExpansion, rm1_expansion, rm3_expansion

# Initialize RM expansion
rm = RMExpansion()

# Sample data
query = "machine learning algorithms"
documents = [
    "Machine learning algorithms are used in AI applications.",
    "Neural networks process data using learning algorithms.",
    "Algorithms in computer science solve computational problems."
]
scores = [0.9, 0.8, 0.7]  # Pseudo-relevance scores

# RM3 expansion (includes original query terms)
expansion_terms = rm.expand_query(query, documents, scores, num_expansion_terms=10, rm_type="rm3")

# RM1 expansion (excludes original query terms)
expansion_terms = rm.expand_query(query, documents, scores, num_expansion_terms=10, rm_type="rm1")
```

## API Reference

### RMExpansion Class

#### Constructor

```python
RMExpansion(
    stopwords=None,           # Custom stopwords set
    min_term_length=2,        # Minimum term length
    max_term_length=50,       # Maximum term length
    remove_query_terms=False, # Remove original query terms
    use_nltk_stopwords=True,  # Use NLTK stopwords
    language='english'        # Language for NLTK stopwords
)
```

#### Main Methods

**expand_query()**
```python
expand_query(
    query: str,                    # Original query
    documents: List[str],          # Pseudo-relevant documents
    scores: List[float],           # Document relevance scores
    num_expansion_terms: int = 10, # Number of expansion terms
    rm_type: str = "rm3"          # "rm1" or "rm3"
) -> List[Tuple[str, float]]      # Returns [(term, weight), ...]
```

**compute_term_weights()**
```python
compute_term_weights(
    documents: List[str],
    scores: List[float],
    query_terms: List[str] = None
) -> Dict[str, float]             # Returns {term: weight}
```

## Configuration Options

### Stopword Handling

```python
# Use NLTK stopwords (recommended)
rm = RMExpansion(use_nltk_stopwords=True, language='english')

# Custom stopwords
custom_stopwords = {'machine', 'learning', 'algorithm'}
rm = RMExpansion(stopwords=custom_stopwords, use_nltk_stopwords=False)

# No stopword filtering
rm = RMExpansion(use_nltk_stopwords=False, stopwords=set())
```

### Term Filtering

```python
# Custom term length constraints
rm = RMExpansion(
    min_term_length=3,    # Minimum 3 characters
    max_term_length=15,   # Maximum 15 characters
    remove_query_terms=True  # Remove original query terms
)
```

## RM1 vs RM3

### RM1 (Traditional Relevance Model)
- **Purpose**: Add new terms not in the original query
- **Behavior**: Filters out original query terms
- **Use case**: Pure expansion without query term emphasis

```python
rm1_terms = rm1_expansion(query, documents, scores, num_terms=10)
# Returns only new expansion terms
```

### RM3 (Mixed Relevance Model)
- **Purpose**: Combine original query with expansion terms
- **Behavior**: Includes original query terms with high weights
- **Use case**: Query enhancement with term re-weighting

```python
rm3_terms = rm3_expansion(query, documents, scores, num_terms=10)
# Returns original query terms + expansion terms
```

## Advanced Usage

### Integration with Multi-Vector Retrieval

```python
from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity

# Initialize components
rm = RMExpansion()
semantic_sim = SemanticSimilarity()

# Get expansion terms
expansion_terms = rm.expand_query(query, pseudo_docs, scores, rm_type="rm3")

# Compute semantic similarities for importance weighting
expansion_words = [term for term, weight in expansion_terms]
semantic_scores = semantic_sim.compute_query_expansion_similarities(query, expansion_words)

# Combine RM weights with semantic scores
for term, rm_weight in expansion_terms:
    semantic_score = semantic_scores.get(term, 0.0)
    combined_importance = rm_weight + semantic_score
    # Use combined_importance for multi-vector scaling
```

### Batch Processing

```python
# Process multiple queries efficiently
queries = ["query1", "query2", "query3"]
all_expansion_terms = {}

for query_id, query_text in enumerate(queries):
    pseudo_docs = get_pseudo_relevant_docs(query_text)
    scores = get_relevance_scores(query_text, pseudo_docs)
    
    expansion_terms = rm.expand_query(query_text, pseudo_docs, scores)
    all_expansion_terms[query_id] = expansion_terms
```

### Statistics and Analysis

```python
# Analyze expansion quality
expansion_terms = rm.expand_query(query, documents, scores)

# Weight distribution
weights = [weight for term, weight in expansion_terms]
print(f"Weight range: {min(weights):.3f} - {max(weights):.3f}")
print(f"Average weight: {sum(weights)/len(weights):.3f}")

# Term characteristics
terms = [term for term, weight in expansion_terms]
avg_length = sum(len(term) for term in terms) / len(terms)
print(f"Average term length: {avg_length:.1f}")
```

## Performance Considerations

### Optimization Tips

1. **Limit Document Length**: Truncate very long documents to improve speed
2. **Term Filtering**: Use appropriate min/max term lengths to reduce noise
3. **Stopword Removal**: Essential for removing common, uninformative terms
4. **Score Normalization**: Ensure relevance scores are properly normalized

### Memory Usage

```python
# For large document collections, process in batches
def process_large_collection(query, documents, scores, batch_size=100):
    all_terms = []
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_scores = scores[i:i+batch_size]
        
        batch_terms = rm.expand_query(query, batch_docs, batch_scores)
        all_terms.extend(batch_terms)
    
    # Merge and re-rank terms
    return merge_and_rerank_terms(all_terms)
```

## Common Issues and Solutions

### Issue: Poor Expansion Quality

**Symptoms**: Expansion terms are irrelevant or too generic

**Solutions**:
```python
# Increase minimum term length
rm = RMExpansion(min_term_length=4)

# Use more restrictive stopwords
rm = RMExpansion(use_nltk_stopwords=True)

# Filter by term frequency
def filter_by_frequency(expansion_terms, min_freq=2):
    # Filter terms that appear in multiple documents
    return [(term, weight) for term, weight in expansion_terms 
            if document_frequency(term) >= min_freq]
```

### Issue: No Expansion Terms Generated

**Symptoms**: `expand_query()` returns empty list

**Solutions**:
```python
# Check input data
assert len(documents) > 0, "No documents provided"
assert len(scores) > 0, "No scores provided"
assert all(score > 0 for score in scores), "All scores are zero"

# Reduce filtering constraints
rm = RMExpansion(min_term_length=1, max_term_length=100)

# Check for very restrictive stopwords
print(f"Stopwords count: {len(rm.stopwords)}")
```

### Issue: Memory Issues with Large Collections

**Solutions**:
```python
# Process in smaller batches
rm = RMExpansion()
expansion_terms = rm.expand_query(
    query, 
    documents[:1000],  # Limit to top 1000 documents
    scores[:1000], 
    num_expansion_terms=10
)

# Use text preprocessing to reduce size
def preprocess_documents(docs):
    return [doc[:500] for doc in docs]  # Truncate to 500 chars
```

## Testing

Run RM expansion tests:

```bash
# Run all RM expansion tests
python -m pytest tests/test_rm_expansion.py -v

# Run specific test
python tests/test_rm_expansion.py --test test_basic_rm3_expansion

# Run with different configurations
python -c "
from tests.test_rm_expansion import TestRMExpansion
test = TestRMExpansion()
test.setUp()
test.test_basic_rm3_expansion()
print('RM expansion test passed!')
"
```

