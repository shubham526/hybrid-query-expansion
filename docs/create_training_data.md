# Training Data Creation Guide

## Overview

The `create_training_data.py` script extracts features for importance weight learning from MSMARCO and other IR datasets. It combines RM expansion, BM25 scores, and semantic similarity to create training data for the weight optimization process.

## Quick Start

```bash
# Basic usage (no BM25 index)
python scripts/create_training_data.py \
    --output_dir ./training_data \
    --max_queries 1000

# With BM25 index (recommended)
python scripts/create_training_data.py \
    --output_dir ./training_data \
    --index_path ./indexes/msmarco-passage_bert-base-uncased \
    --lucene_path /path/to/lucene/* \
    --max_queries 10000
```

## Command Line Arguments

### Required Arguments

- `--output_dir`: Output directory for training data
- `--lucene_path`: Path to Lucene JAR files (required if using BM25 index)

### Optional Arguments

**Data Parameters:**
- `--dataset`: IR dataset name (default: `msmarco-passage/train/triples-small`)
- `--max_queries`: Maximum queries to process (default: all)
- `--index_path`: Path to BM25 index (optional)

**Model Parameters:**
- `--semantic_model`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `--max_expansion_terms`: Max expansion terms per query (default: 15)
- `--min_relevant_docs`: Min relevant docs required (default: 1)

**Logging:**
- `--log_level`: Logging level (DEBUG/INFO/WARNING/ERROR)

## Output Structure

The script creates a structured training dataset:

```
training_data/
├── queries.json              # Query collection
├── qrels.json                # Relevance judgments
├── documents.pkl.gz          # Document collection (compressed)
├── features.json.gz          # Main training features (compressed)
├── statistics.json           # Feature statistics
└── metadata.json             # Experiment metadata
```

### Feature Format

Each query in `features.json.gz` has this structure:

```json
{
    "query_id": "12345",
    "query_text": "machine learning algorithms",
    "expansion_terms": [["neural", 0.8], ["networks", 0.6]],
    "term_features": {
        "neural": {
            "rm_weight": 0.8,
            "bm25_score": 2.1,
            "semantic_score": 0.75
        },
        "networks": {
            "rm_weight": 0.6,
            "bm25_score": 1.8,
            "semantic_score": 0.68
        }
    },
    "num_relevant_docs": 3,
    "reference_doc_id": "doc_123"
}
```

## Usage Examples

### Development and Testing

```bash
# Small dataset for quick testing
python scripts/create_training_data.py \
    --output_dir ./test_data \
    --max_queries 100 \
    --log_level DEBUG

# No BM25 index (semantic + RM only)
python scripts/create_training_data.py \
    --output_dir ./semantic_only_data \
    --semantic_model all-mpnet-base-v2 \
    --max_queries 1000
```

### Production Data Creation

```bash
# Full MSMARCO training data
python scripts/create_training_data.py \
    --output_dir ./msmarco_full_training \
    --dataset msmarco-passage/train/triples-small \
    --index_path ./indexes/msmarco-passage_bert-base-uncased \
    --lucene_path /opt/lucene/jars/* \
    --semantic_model all-mpnet-base-v2 \
    --max_expansion_terms 20

# Different dataset (TREC CAR)
python scripts/create_training_data.py \
    --output_dir ./trec_car_training \
    --dataset trec-car/v1.5/train/fold0 \
    --max_queries 5000
```

### Custom Model Experiments

```bash
# Different semantic models
for model in all-MiniLM-L6-v2 all-mpnet-base-v2 msmarco-distilbert-base-v4; do
    python scripts/create_training_data.py \
        --output_dir ./training_data_${model//\//_} \
        --semantic_model $model \
        --max_queries 5000
done

# Different expansion term counts
for num_terms in 10 15 20 25; do
    python scripts/create_training_data.py \
        --output_dir ./training_data_${num_terms}_terms \
        --max_expansion_terms $num_terms \
        --max_queries 5000
done
```

## Feature Extraction Process

### 1. Data Loading

```python
# Loads from ir_datasets
dataset = ir_datasets.load("msmarco-passage/train/triples-small")
queries = {q.query_id: q.text for q in dataset.queries_iter()}
documents = {d.doc_id: d.text for d in dataset.docs_iter()}
qrels = {qrel.query_id: {qrel.doc_id: qrel.relevance} for qrel in dataset.qrels_iter()}
```

### 2. For Each Query

**Step 1: Get Relevant Documents**
```python
relevant_docs = [doc_id for doc_id, rel in qrels[query_id].items() if rel >= 1]
```

**Step 2: RM Expansion**
```python
pseudo_docs = [documents[doc_id] for doc_id in relevant_docs]
expansion_terms = rm_expansion.expand_query(query_text, pseudo_docs, scores)
```

**Step 3: Feature Computation**
```python
for term, rm_weight in expansion_terms:
    # BM25 score
    bm25_score = bm25_scorer.compute_bm25_term_weight(reference_doc_id, [term])[term]
    
    # Semantic similarity
    semantic_score = semantic_sim.compute_similarity(term, query_text)
    
    # Store features
    term_features[term] = {
        'rm_weight': rm_weight,
        'bm25_score': bm25_score, 
        'semantic_score': semantic_score
    }
```

## Loading and Using Training Data

### In Python Scripts

```python
from src.utils.file_utils import load_training_data

# Load training data
training_data = load_training_data('./training_data')

# Access components
queries = training_data['queries']
features = training_data['features']
statistics = training_data['statistics']

# Process features for weight learning
all_features = []
for query_id, query_features in features.items():
    for term, term_data in query_features['term_features'].items():
        feature_vector = [
            term_data['rm_weight'],
            term_data['bm25_score'],
            term_data['semantic_score']
        ]
        all_features.append(feature_vector)
```

### For Weight Training

```python
# Use in weight training script
python scripts/train_weights.py \
    --training_data ./training_data \
    --output_dir ./models
```

## Performance and Optimization

### Speed Optimization

**Parallel Processing:**
```bash
# Use multiple processes for large datasets
export OMP_NUM_THREADS=4
python scripts/create_training_data.py \
    --output_dir ./training_data \
    --max_queries 50000
```

**Memory Optimization:**
```bash
# Process in smaller batches for memory efficiency
python scripts/create_training_data.py \
    --output_dir ./training_data \
    --max_queries 10000  # Process in chunks
```

**Storage Optimization:**
```bash
# Use compression for large datasets
python scripts/create_training_data.py \
    --output_dir ./training_data \
    --max_queries 100000  # Automatically compresses large files
```

### Hardware Requirements

**Minimum Requirements:**
- RAM: 8GB (for datasets up to 10K queries)
- Storage: 5GB free space
- CPU: Multi-core recommended

**Recommended for Large Datasets:**
- RAM: 32GB+ (for full MSMARCO)
- Storage: 50GB+ free space
- GPU: For faster sentence transformer inference

## Troubleshooting

### Common Issues

**1. Memory Errors**
```bash
# Reduce batch size
python scripts/create_training_data.py \
    --output_dir ./training_data \
    --max_queries 5000  # Smaller dataset

# Monitor memory usage
htop  # Check RAM usage during processing
```

**2. Lucene Initialization Fails**
```bash
# Check Java installation
java -version

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk

# Use correct Lucene path
--lucene_path /path/to/lucene/jars/*  # Note the /* at the end
```

**3. Slow Processing**
```bash
# Use faster semantic model
--semantic_model all-MiniLM-L6-v2  # Faster than all-mpnet-base-v2

# Reduce expansion terms
--max_expansion_terms 10  # Instead of 20

# Process smaller batches
--max_queries 1000
```

**4. No Features Generated**
```bash
# Check dataset availability
python -c "import ir_datasets; print(ir_datasets.load('msmarco-passage/train/triples-small').queries_count())"

# Reduce minimum relevant docs requirement
--min_relevant_docs 1

# Check logs for errors
--log_level DEBUG
```

### Validation

**Check Training Data Quality:**
```python
from src.utils.file_utils import load_training_data
import json

# Load and inspect
training_data = load_training_data('./training_data')

# Check statistics
stats = training_data['statistics']
print(f"Processed queries: {stats['num_queries']}")
print(f"Avg expansion terms: {stats['avg_expansion_terms']:.1f}")
print(f"Feature stats: {json.dumps(stats['feature_stats'], indent=2)}")

# Check feature distribution
features = training_data['features']
rm_weights = []
bm25_scores = []
semantic_scores = []

for query_features in features.values():
    for term_data in query_features['term_features'].values():
        rm_weights.append(term_data['rm_weight'])
        bm25_scores.append(term_data['bm25_score'])
        semantic_scores.append(term_data['semantic_score'])

print(f"RM weights range: {min(rm_weights):.3f} - {max(rm_weights):.3f}")
print(f"BM25 scores range: {min(bm25_scores):.3f} - {max(bm25_scores):.3f}")
print(f"Semantic scores range: {min(semantic_scores):.3f} - {max(semantic_scores):.3f}")
```

## Integration with Weight Training

Once training data is created, use it for weight learning:

```bash
# Create training data
python scripts/create_training_data.py \
    --output_dir ./training_data \
    --index_path ./indexes/msmarco-passage_bert-base-uncased \
    --lucene_path /path/to/lucene/* \
    --max_queries 10000

# Train weights using the data
python scripts/train_weights.py \
    --training_data ./training_data \
    --output_dir ./models \
    --index_path ./indexes/msmarco-passage_bert-base-uncased \
    --lucene_path /path/to/lucene/*

# Evaluate the trained model
python scripts/evaluate_model.py \
    --weights_file ./models/learned_weights.json \
    --dataset msmarco-passage/trec-dl-2019 \
    --output_dir ./results
```

## Advanced Configuration

### Custom Feature Extraction

For custom feature extraction, modify the `TrainingDataCreator` class:

```python
from scripts.create_training_data import TrainingDataCreator

class CustomTrainingDataCreator(TrainingDataCreator):
    def extract_features_for_query(self, query_id, query_text, relevant_docs, documents):
        # Call parent method
        features = super().extract_features_for_query(query_id, query_text, relevant_docs, documents)
        
        # Add custom features
        if 'term_features' in features:
            for term, term_data in features['term_features'].items():
                # Add custom feature
                term_data['custom_score'] = compute_custom_score(term, query_text)
        
        return features
```

### Different Datasets

```bash
# TREC Robust
python scripts/create_training_data.py \
    --output_dir ./robust04_training \
    --dataset robust04 \
    --max_queries 1000

# MS MARCO Document
python scripts/create_training_data.py \
    --output_dir ./msmarco_doc_training \
    --dataset msmarco-document/train \
    --max_queries 5000
```

This comprehensive guide should help you effectively use the training data creation script for your expansion weight learning experiments!