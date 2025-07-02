# BM25 Integration Documentation

## Overview

The BM25 integration provides BERT-tokenized BM25 scoring for importance weight learning. It uses Apache Lucene for indexing and retrieval, with custom tokenization alignment to match BERT subword tokenization. The integration consists of indexing (via `BERTTokenBM25Indexer`) and scoring (via `TokenBM25Scorer`) components.

## Architecture

Based on the actual codebase components:

1. **`BERTTokenBM25Indexer`** (`src/core/bm25_scorer.py`): Creates Lucene indexes with BERT tokenization
2. **`TokenBM25Scorer`** (`src/core/bm25_scorer.py`): Computes BM25 scores for terms in documents
3. **Lucene Integration** (`src/utils/initialize_lucene.py`): JVM and classpath setup
4. **Index Creation Script** (`scripts/create_index.py`): Command-line index creation

## Prerequisites

### Java Requirements

```bash
# Install Java 11 or higher
sudo apt-get update
sudo apt-get install openjdk-11-jdk

# Verify installation
java -version
# Should show: openjdk version "11.0.x" or higher

# Set JAVA_HOME (add to ~/.bashrc)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### Lucene Setup

Download Apache Lucene 10.1.0 JAR files:

```bash
# Create directory for Lucene
mkdir -p /opt/lucene/jars
cd /opt/lucene/jars

# Download required JAR files
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/10.1.0/lucene-core-10.1.0.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-common/10.1.0/lucene-analysis-common-10.1.0.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-queryparser/10.1.0/lucene-queryparser-10.1.0.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-memory/10.1.0/lucene-memory-10.1.0.jar

# Verify files
ls -la *.jar
```

### Python Dependencies

```bash
# Install required packages
pip install pyjnius transformers torch

# Verify PyJNIus installation
python -c "import jnius; print('PyJNIus installed successfully')"
```

## Index Creation

### Using the create_index.py Script

```bash
# Basic index creation
python scripts/create_index.py \
    --collection msmarco-passage \
    --output_dir ./indexes \
    --lucene_path /opt/lucene/jars/*

# Full index creation with custom parameters
python scripts/create_index.py \
    --collection msmarco-passage \
    --model_name bert-base-uncased \
    --output_dir ./indexes \
    --lucene_path /opt/lucene/jars/* \
    --k1 1.2 \
    --b 0.75 \
    --max_length 512 \
    --validate
```

### Manual Index Creation

```python
from src.core.bm25_scorer import BERTTokenBM25Indexer
from src.utils.lucene_utils import initialize_lucene

# Initialize Lucene JVM
lucene_path = "/opt/lucene/jars/*"
if not initialize_lucene(lucene_path):
    raise RuntimeError("Failed to initialize Lucene")

# Load document collection
import ir_datasets

dataset = ir_datasets.load("msmarco-passage")
documents = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

# Create indexer
indexer = BERTTokenBM25Indexer(
    model_name="bert-base-uncased",
    index_path="./indexes/msmarco-passage_bert-base-uncased"
)

# Create index
indexer.create_index(documents)
print(f"Indexed {len(documents)} documents")
```

## BM25 Scoring

### Basic Usage

```python
from src.core.bm25_scorer import TokenBM25Scorer
from src.utils.lucene_utils import initialize_lucene

# Initialize Lucene
initialize_lucene("/opt/lucene/jars/*")

# Create scorer
scorer = TokenBM25Scorer(
    index_path="./indexes/msmarco-passage_bert-base-uncased",
    k1=1.2,
    b=0.75
)

# Compute BM25 scores for terms in a document
doc_id = "7067032"
terms = ["machine", "learning", "algorithm"]

term_scores = scorer.compute_bm25_term_weight(doc_id, terms)
print(term_scores)
# Output: {"machine": 2.156, "learning": 1.843, "algorithm": 0.0}
```

### Token-Level Scoring

```python
# Get BM25 scores for each token position
query = "machine learning algorithms"
doc_id = "7067032"
max_length = 512

token_scores = scorer.get_token_scores(query, doc_id, max_length)
print(f"Token scores shape: {token_scores.shape}")  # torch.Size([512])
print(f"Non-zero scores: {torch.nonzero(token_scores).numel()}")
```

## Integration with Weight Learning

### In Training Pipeline

```python
# train_weights.py integration (actual implementation)
try:
    from src.core.bm25_scorer import TokenBM25Scorer
    from src.utils.lucene_utils import initialize_lucene

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def initialize_bm25_scorer(index_path, lucene_path):
    """Initialize BM25 scorer if available."""
    if not BM25_AVAILABLE:
        return None

    if not initialize_lucene(lucene_path):
        raise RuntimeError("Failed to initialize Lucene")

    return TokenBM25Scorer(index_path)


# Usage in training
if args.index_path and args.lucene_path:
    bm25_scorer = initialize_bm25_scorer(args.index_path, args.lucene_path)
else:
    bm25_scorer = None
    print("No BM25 index provided. BM25 scores will default to 0.0")
```

### In Importance Weight Computation

```python
def compute_importance_weights(query_text, expansion_terms, reference_doc_id, 
                              rm_expansion, semantic_sim, bm25_scorer, 
                              alpha, beta, gamma):
    """Compute importance weights combining RM, BM25, and semantic scores."""
    
    importance_weights = {}
    
    # Get semantic similarities (batch computation)
    expansion_words = [term for term, weight in expansion_terms]
    semantic_scores = semantic_sim.compute_query_expansion_similarities(
        query_text, expansion_words
    )
    
    for term, rm_weight in expansion_terms:
        # BM25 score
        bm25_score = 0.0
        if bm25_scorer and reference_doc_id:
            try:
                bm25_scores = bm25_scorer.compute_bm25_term_weight(reference_doc_id, [term])
                bm25_score = float(bm25_scores.get(term, 0.0))
            except Exception:
                bm25_score = 0.0
        
        # Semantic score
        semantic_score = semantic_scores.get(term, 0.0)
        
        # Combine using learned weights
        importance = alpha * rm_weight + beta * bm25_score + gamma * semantic_score
        importance_weights[term] = importance
    
    return importance_weights
```

## Configuration

### Lucene Initialization

The `initialize_lucene.py` module handles JVM setup:

```python
def initialize_lucene(lucene_path: str) -> bool:
    """Initialize Lucene with proper classpath settings."""
    try:
        import jnius_config
        
        # Required JAR files
        required_jars = [
            'lucene-core-10.1.0.jar',
            'lucene-analysis-common-10.1.0.jar', 
            'lucene-queryparser-10.1.0.jar',
            'lucene-memory-10.1.0.jar'
        ]
        
        # Build classpath
        jar_paths = []
        for jar in required_jars:
            full_path = os.path.join(lucene_path, jar)
            if not os.path.exists(full_path):
                raise ValueError(f"Required JAR not found: {full_path}")
            jar_paths.append(full_path)
        
        # Set JVM options and classpath
        jnius_config.add_options('-Xmx4096m', '-Xms1024m')
        jnius_config.set_classpath(os.pathsep.join(jar_paths))
        
        # Test class loading
        from jnius import autoclass
        FSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Lucene: {e}")
        return False
```

### BM25 Parameters

```python
# Default parameters (from codebase)
scorer = TokenBM25Scorer(
    index_path="./indexes/msmarco-passage_bert-base-uncased",
    k1=1.2,      # Term frequency saturation parameter
    b=0.75       # Length normalization parameter
)

# Custom parameters for different collections
# For longer documents (papers, books):
scorer = TokenBM25Scorer(index_path, k1=1.6, b=0.75)

# For shorter documents (tweets, titles):  
scorer = TokenBM25Scorer(index_path, k1=0.9, b=0.4)
```

## BERT Tokenization Alignment

### How It Works

The `BERTTokenBM25Indexer` aligns BERT subword tokens with Lucene indexing:

```python
def create_index(self, documents: Dict[str, str]):
    """Create index with BERT token position mapping."""
    
    for doc_id, doc_text in documents.items():
        # BERT tokenization
        tokens = self.tokenizer.tokenize(doc_text)
        
        # Build full words and track positions
        words = []
        word_positions = []  # (word_index, start_token_pos, end_token_pos)
        
        current_word = ""
        word_start_pos = 0
        
        for i, token in enumerate(tokens):
            if token.startswith('##'):
                # Continuation of previous word
                current_word += token[2:]
            else:
                # New word starts
                if current_word:
                    words.append(current_word)
                    word_positions.append((len(words) - 1, word_start_pos, i - 1))
                current_word = token
                word_start_pos = i
        
        # Add last word
        if current_word:
            words.append(current_word)
            word_positions.append((len(words) - 1, word_start_pos, len(tokens) - 1))
        
        # Create Lucene document
        lucene_doc = self.Document()
        lucene_doc.add(self.StringField("id", doc_id, self.FieldStore.YES))
        lucene_doc.add(self.TextField("contents", " ".join(words), self.FieldStore.YES))
        lucene_doc.add(self.StoredField("positions", json.dumps(word_positions)))
        
        writer.addDocument(lucene_doc)
```

### Token Score Mapping

```python
def get_token_scores(self, query: str, doc_id: str, max_length: int = 512):
    """Map BM25 scores to BERT token positions."""
    
    # Get document and position information
    doc = self.searcher.storedFields().document(doc_hits.scoreDocs[0].doc)
    doc_content = doc.get("contents")
    positions_str = doc.get("positions")
    
    word_positions = json.loads(positions_str)
    doc_words = doc_content.split()
    
    # Get BM25 scores for query terms
    query_terms = query.lower().split()
    term_scores = self.compute_bm25_term_weight(doc_id, query_terms)
    
    # Map to token positions
    token_scores = np.zeros(max_length)
    
    for word_idx, start_pos, end_pos in word_positions:
        if 0 <= word_idx < len(doc_words):
            word = doc_words[word_idx].lower()
            score = term_scores.get(word, 0.0)
            
            if score > 0:
                # Apply score to all subword tokens of this word
                end_pos = min(end_pos, max_length - 1)
                token_scores[start_pos:end_pos + 1] = score
    
    return torch.FloatTensor(token_scores)
```

## Performance Optimization

### Indexing Performance

```bash
# Monitor indexing progress
python scripts/create_index.py \
    --collection msmarco-passage \
    --output_dir ./indexes \
    --lucene_path /opt/lucene/jars/* \
    --log_level DEBUG

# Parallel processing (if available)
export OMP_NUM_THREADS=8
python scripts/create_index.py \
    --collection msmarco-passage \
    --output_dir ./indexes \
    --lucene_path /opt/lucene/jars/*
```

### Scoring Performance

```python
# Batch term scoring for better performance
terms = ["machine", "learning", "neural", "network", "algorithm"]
scores = scorer.compute_bm25_term_weight(doc_id, terms)

# Rather than individual calls:
# scores = {term: scorer.compute_bm25_term_weight(doc_id, [term])[term] for term in terms}
```

### Memory Management

```python
# For large collections, monitor memory usage
import psutil
import gc

def check_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Clean up after large operations
del large_objects
gc.collect()
```

## Troubleshooting

### Common Issues

**1. Java/JVM Issues**

```bash
# Error: JAVA_HOME not set
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Error: Java version incompatible
java -version  # Should be 11+

# Error: OutOfMemoryError
# Increase JVM heap size in initialize_lucene.py:
jnius_config.add_options('-Xmx8192m', '-Xms2048m')
```

**2. PyJNIus Installation Issues**

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev default-jdk

# Install PyJNIus with proper Java path
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
pip install pyjnius

# Verify installation
python -c "import jnius; print('Success')"
```

**3. Lucene JAR File Issues**

```bash
# Error: Required JAR not found
# Check JAR files exist:
ls -la /opt/lucene/jars/
# Should show all 4 required JAR files

# Download missing JARs:
cd /opt/lucene/jars/
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/10.1.0/lucene-core-10.1.0.jar
```

**4. Index Creation Failures**

```python
# Debug index creation
try:
    indexer.create_index(documents)
except Exception as e:
    print(f"Indexing failed: {e}")
    import traceback
    traceback.print_exc()

# Check disk space
import shutil
free_space = shutil.disk_usage('.').free / (1024**3)
print(f"Free disk space: {free_space:.1f} GB")
```

**5. BM25 Scoring Issues**

```python
# Debug term scoring
term_scores = scorer.compute_bm25_term_weight(doc_id, ["test"])
if not term_scores or all(score == 0.0 for score in term_scores.values()):
    print("No BM25 scores returned - check:")
    print("1. Document exists in index")
    print("2. Terms are properly analyzed")
    print("3. Index was created successfully")
```

### Debug Mode

```python
# Enable debug logging for BM25 components
import logging

logging.basicConfig(level=logging.DEBUG)

# Test basic functionality
from src.core.bm25_scorer import TokenBM25Scorer
from src.utils.lucene_utils import initialize_lucene

# Step-by-step debugging
print("1. Initializing Lucene...")
success = initialize_lucene("/opt/lucene/jars/*")
print(f"   Success: {success}")

if success:
    print("2. Creating scorer...")
    scorer = TokenBM25Scorer("./indexes/msmarco-passage_bert-base-uncased")
    print("   Scorer created successfully")

    print("3. Testing term scoring...")
    scores = scorer.compute_bm25_term_weight("7067032", ["machine"])
    print(f"   Scores: {scores}")
```

## Index Validation

### Validation Script

```python
def validate_bm25_index(index_path, sample_queries=None):
    """Validate BM25 index functionality."""
    
    if sample_queries is None:
        sample_queries = [
            "machine learning",
            "neural networks", 
            "information retrieval"
        ]
    
    try:
        scorer = TokenBM25Scorer(index_path)
        
        print("BM25 Index Validation")
        print("=" * 30)
        
        for query in sample_queries:
            # Test term scoring
            terms = query.split()
            scores = scorer.compute_bm25_term_weight("7067032", terms)  # Sample doc ID
            
            print(f"Query: '{query}'")
            for term, score in scores.items():
                print(f"  {term}: {score:.4f}")
            print()
        
        print("Validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

# Run validation
validate_bm25_index("./indexes/msmarco-passage_bert-base-uncased")
```

## Best Practices

### 1. Index Management

```bash
# Create indexes with descriptive names
./indexes/
├── msmarco-passage_bert-base-uncased/     # Standard BERT
├── msmarco-passage_distilbert-base/       # DistilBERT
└── msmarco-doc_bert-base-uncased/         # Document collection
```

### 2. Parameter Tuning

```python
# Test different BM25 parameters for your collection
k1_values = [0.9, 1.2, 1.6, 2.0]
b_values = [0.4, 0.75, 1.0]

for k1 in k1_values:
    for b in b_values:
        scorer = TokenBM25Scorer(index_path, k1=k1, b=b)
        # Evaluate performance...
```

### 3. Error Handling

```python
# Robust BM25 integration
def safe_bm25_scoring(bm25_scorer, doc_id, terms):
    """Safely compute BM25 scores with fallback."""
    try:
        if bm25_scorer is None:
            return {term: 0.0 for term in terms}
        
        scores = bm25_scorer.compute_bm25_term_weight(doc_id, terms)
        return scores if scores else {term: 0.0 for term in terms}
        
    except Exception as e:
        logger.warning(f"BM25 scoring failed for doc {doc_id}: {e}")
        return {term: 0.0 for term in terms}
```

### 4. Resource Management

```python
# Proper cleanup for long-running processes
class BM25Manager:
    def __init__(self, index_path):
        self.scorer = TokenBM25Scorer(index_path)
    
    def __enter__(self):
        return self.scorer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        if hasattr(self.scorer, 'reader'):
            self.scorer.reader.close()

# Usage
with BM25Manager("./indexes/msmarco-passage_bert-base-uncased") as scorer:
    scores = scorer.compute_bm25_term_weight(doc_id, terms)
```
