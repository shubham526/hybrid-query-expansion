# Model Evaluation Documentation

## Overview

The evaluation framework provides tools for evaluating importance-weighted query expansion models using the existing components in the codebase. It leverages the `TRECEvaluator` and `ExpansionEvaluator` classes along with `pytrec_eval` for standard IR metrics computation.

## Available Components

Based on the codebase, the evaluation framework consists of:

1. **TRECEvaluator** (`src/evaluation/evaluator.py`): Standard TREC-style evaluation
2. **ExpansionEvaluator** (`src/evaluation/evaluator.py`): Specialized evaluation for query expansion
3. **Metrics Module** (`src/evaluation/metrics.py`): Wrapper around pytrec_eval
4. **TRECDLReranker** (`src/models/multivector_reranking.py`): Multi-vector reranking for evaluation

## Core Evaluation Workflow

### Using the TRECEvaluator

```python
from src.evaluation.evaluator import TRECEvaluator, create_trec_dl_evaluator

# Initialize evaluator
evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10', 'ndcg_cut_100', 'recip_rank'])

# Evaluate a single run
run_results = {
    'query1': [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)],
    'query2': [('doc4', 0.95), ('doc1', 0.85)]
}

qrels = {
    'query1': {'doc1': 2, 'doc2': 1, 'doc3': 0},
    'query2': {'doc4': 3, 'doc1': 1}
}

results = evaluator.evaluate_run(run_results, qrels)
print(f"MAP: {results['map']:.4f}")
print(f"nDCG@10: {results['ndcg_cut_10']:.4f}")
```

### Comparing Multiple Models

```python
# Compare multiple expansion approaches
runs = {
    'baseline': baseline_run_results,
    'rm_only': rm_only_results, 
    'our_method': our_method_results
}

comparison = evaluator.compare_runs(runs, qrels, baseline_run='baseline')

# Print comparison table
table = evaluator.create_results_table(comparison)
print(table)
```

## TREC DL Evaluation Pipeline

### Using TRECDLReranker for Evaluation

```python
from src.models.multivector_reranking import TRECDLReranker, create_trec_dl_evaluation_pipeline

# Load TREC DL data
import ir_datasets
from collections import defaultdict

dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")

# Load components
queries = {q.query_id: q.text for q in dataset.queries_iter()}
qrels = defaultdict(dict)
for qrel in dataset.qrels_iter():
    qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

first_stage_runs = defaultdict(list)
for scoreddoc in dataset.scoreddocs_iter():
    first_stage_runs[scoreddoc.query_id].append((scoreddoc.doc_id, scoreddoc.score))

# Initialize reranker
reranker = create_trec_dl_evaluation_pipeline("all-MiniLM-L6-v2", "2019")

# Compute expansion terms and importance weights (your pipeline)
expansion_terms_dict = {}  # query_id -> [(term, rm_weight), ...]
importance_weights_dict = {}  # query_id -> {term: importance_score}

# Your expansion computation here...
for query_id, query_text in queries.items():
    # RM expansion
    expansion_terms = rm_expansion.expand_query(query_text, pseudo_docs, scores)
    
    # Compute importance weights using learned weights (α, β, γ)
    importance_weights = {}
    for term, rm_weight in expansion_terms:
        bm25_score = bm25_scorer.compute_bm25_term_weight(ref_doc_id, [term])[term] if bm25_scorer else 0.0
        semantic_score = semantic_sim.compute_similarity(term, query_text)
        importance = alpha * rm_weight + beta * bm25_score + gamma * semantic_score
        importance_weights[term] = importance
    
    expansion_terms_dict[query_id] = expansion_terms
    importance_weights_dict[query_id] = importance_weights

# Rerank using importance weights
reranked_results = reranker.rerank_trec_dl_run(
    queries=queries,
    first_stage_runs=dict(first_stage_runs),
    expansion_terms_dict=expansion_terms_dict,
    importance_weights_dict=importance_weights_dict,
    top_k=100
)

# Evaluate improvements
evaluation_results = reranker.evaluate_reranking(
    original_runs=dict(first_stage_runs),
    reranked_runs=reranked_results,
    qrels=dict(qrels),
    metrics=['ndcg_cut_10', 'ndcg_cut_100', 'map']
)

print("Evaluation Results:")
for metric in ['ndcg_cut_10', 'map']:
    original = evaluation_results['original'][metric]
    reranked = evaluation_results['reranked'][metric]
    improvement = evaluation_results['improvement'][metric]
    print(f"{metric.upper()}: {original:.4f} → {reranked:.4f} (+{improvement:.4f})")
```

## Available Metrics

The `get_metric` function in `src/evaluation/metrics.py` supports all pytrec_eval metrics:

```python
from src.evaluation.metrics import get_metric

# Primary TREC DL metrics
metrics = [
    'map',              # Mean Average Precision
    'ndcg_cut_10',      # nDCG@10 (primary TREC DL metric)
    'ndcg_cut_100',     # nDCG@100
    'recip_rank',       # Mean Reciprocal Rank
    'recall_100',       # Recall@100
    'P_10',             # Precision@10
]

# Evaluate using temporary TREC files
import tempfile
import os

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.qrel') as qrel_file, \
     tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run') as run_file:
    
    # Write qrels
    for query_id, docs in qrels.items():
        for doc_id, relevance in docs.items():
            qrel_file.write(f"{query_id} 0 {doc_id} {relevance}\n")
    qrel_file.flush()
    
    # Write run
    for query_id, docs in run_results.items():
        for rank, (doc_id, score) in enumerate(docs, 1):
            run_file.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} run\n")
    run_file.flush()
    
    # Compute metrics
    for metric in metrics:
        score = get_metric(qrel_file.name, run_file.name, metric)
        print(f"{metric}: {score:.4f}")
    
    # Cleanup
    os.unlink(qrel_file.name)
    os.unlink(run_file.name)
```

## Ablation Study Framework

### Manual Ablation Study

```python
def run_ablation_study(queries, qrels, first_stage_runs, 
                      rm_expansion, semantic_sim, bm25_scorer, reranker):
    """Run systematic ablation study."""
    
    # Define ablation configurations
    configs = {
        'rm_only': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0},
        'bm25_only': {'alpha': 0.0, 'beta': 1.0, 'gamma': 0.0},
        'semantic_only': {'alpha': 0.0, 'beta': 0.0, 'gamma': 1.0},
        'rm_bm25': {'alpha': 1.2, 'beta': 0.8, 'gamma': 0.0},
        'rm_semantic': {'alpha': 1.2, 'beta': 0.0, 'gamma': 1.5},
        'our_method': {'alpha': 1.2, 'beta': 0.8, 'gamma': 1.5}
    }
    
    results = {}
    
    for config_name, weights in configs.items():
        print(f"Evaluating {config_name}...")
        
        # Compute importance weights for this configuration
        importance_weights_dict = compute_importance_weights(
            queries, first_stage_runs, rm_expansion, semantic_sim, bm25_scorer,
            alpha=weights['alpha'], beta=weights['beta'], gamma=weights['gamma']
        )
        
        # Rerank
        reranked_results = reranker.rerank_trec_dl_run(
            queries=queries,
            first_stage_runs=first_stage_runs,
            expansion_terms_dict=expansion_terms_dict,
            importance_weights_dict=importance_weights_dict,
            top_k=100
        )
        
        # Evaluate
        evaluator = TRECEvaluator(['ndcg_cut_10', 'map'])
        scores = evaluator.evaluate_run(reranked_results, qrels)
        results[config_name] = scores
    
    return results

def compute_importance_weights(queries, first_stage_runs, rm_expansion, 
                              semantic_sim, bm25_scorer, alpha, beta, gamma):
    """Compute importance weights for given alpha, beta, gamma."""
    importance_weights_dict = {}
    
    for query_id, query_text in queries.items():
        # Get pseudo-relevant documents
        top_docs = first_stage_runs[query_id][:10]
        pseudo_docs = [get_document_text(doc_id) for doc_id, _ in top_docs]
        pseudo_scores = [score for _, score in top_docs]
        
        # RM expansion
        expansion_terms = rm_expansion.expand_query(query_text, pseudo_docs, pseudo_scores)
        
        # Compute importance weights
        importance_weights = {}
        for term, rm_weight in expansion_terms:
            # BM25 score
            bm25_score = 0.0
            if bm25_scorer and top_docs:
                ref_doc_id = top_docs[0][0]  # Use top document as reference
                try:
                    bm25_scores = bm25_scorer.compute_bm25_term_weight(ref_doc_id, [term])
                    bm25_score = bm25_scores.get(term, 0.0)
                except:
                    bm25_score = 0.0
            
            # Semantic score
            semantic_score = semantic_sim.compute_similarity(term, query_text)
            
            # Combine using current weights
            importance = alpha * rm_weight + beta * bm25_score + gamma * semantic_score
            importance_weights[term] = importance
        
        importance_weights_dict[query_id] = importance_weights
    
    return importance_weights_dict

# Run ablation study
ablation_results = run_ablation_study(
    queries, dict(qrels), dict(first_stage_runs),
    rm_expansion, semantic_sim, bm25_scorer, reranker
)

# Print results table
print("Ablation Study Results")
print("=" * 50)
print(f"{'Method':<15} {'nDCG@10':<10} {'MAP':<10}")
print("-" * 35)
for method, scores in ablation_results.items():
    print(f"{method:<15} {scores['ndcg_cut_10']:<10.4f} {scores['map']:<10.4f}")
```

## Integration with Learned Weights

### Loading and Using Learned Weights

```python
from src.utils.file_utils import load_learned_weights

# Load weights from training
weights = load_learned_weights('./models/learned_weights.json')
alpha, beta, gamma = weights

print(f"Learned weights: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")

# Use in evaluation
importance_weights_dict = compute_importance_weights(
    queries, first_stage_runs, rm_expansion, semantic_sim, bm25_scorer,
    alpha=alpha, beta=beta, gamma=gamma
)

# Evaluate performance
reranked_results = reranker.rerank_trec_dl_run(
    queries=queries,
    first_stage_runs=first_stage_runs,
    expansion_terms_dict=expansion_terms_dict,
    importance_weights_dict=importance_weights_dict,
    top_k=100
)

evaluator = create_trec_dl_evaluator("2019")
evaluation = evaluator.evaluate_run(reranked_results, dict(qrels))

print("Final Evaluation Results:")
for metric, score in evaluation.items():
    print(f"  {metric}: {score:.4f}")
```

## Cross-Dataset Evaluation

### Evaluating on Multiple TREC DL Years

```python
def evaluate_cross_dataset(weights_file, datasets):
    """Evaluate learned weights across multiple datasets."""
    
    # Load learned weights
    alpha, beta, gamma = load_learned_weights(weights_file)
    
    results = {}
    
    for dataset_name in datasets:
        print(f"Evaluating on {dataset_name}...")
        
        # Load dataset
        dataset = ir_datasets.load(dataset_name)
        queries = {q.query_id: q.text for q in dataset.queries_iter()}
        
        qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        first_stage_runs = defaultdict(list)
        for scoreddoc in dataset.scoreddocs_iter():
            first_stage_runs[scoreddoc.query_id].append((scoreddoc.doc_id, scoreddoc.score))
        
        # Evaluate using learned weights
        importance_weights_dict = compute_importance_weights(
            queries, dict(first_stage_runs), rm_expansion, semantic_sim, bm25_scorer,
            alpha=alpha, beta=beta, gamma=gamma
        )
        
        reranked_results = reranker.rerank_trec_dl_run(
            queries=queries,
            first_stage_runs=dict(first_stage_runs),
            expansion_terms_dict=expansion_terms_dict,
            importance_weights_dict=importance_weights_dict,
            top_k=100
        )
        
        # Evaluate
        evaluator = TRECEvaluator(['ndcg_cut_10', 'map'])
        scores = evaluator.evaluate_run(reranked_results, dict(qrels))
        results[dataset_name] = scores
    
    return results

# Evaluate on both TREC DL years
datasets = [
    "msmarco-passage/trec-dl-2019",
    "msmarco-passage/trec-dl-2020"
]

cross_dataset_results = evaluate_cross_dataset('./models/learned_weights.json', datasets)

print("Cross-Dataset Results:")
for dataset, scores in cross_dataset_results.items():
    year = dataset.split('-')[-1]
    print(f"TREC DL {year}: nDCG@10={scores['ndcg_cut_10']:.4f}, MAP={scores['map']:.4f}")
```

## Practical Evaluation Script

Here's a complete evaluation script based on the existing codebase:

```python
#!/usr/bin/env python3
"""
Evaluate importance-weighted query expansion model.

Usage:
    python evaluate_model.py \
        --weights_file ./models/learned_weights.json \
        --dataset msmarco-passage/trec-dl-2019 \
        --output_dir ./evaluation_results
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import ir_datasets

from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.evaluation.evaluator import TRECEvaluator
from src.utils.file_utils import load_learned_weights, ensure_dir, save_json
from src.utils.logging_utils import setup_logging

# Import BM25 scorer if available
try:
    from src.core.bm25_scorer import TokenBM25Scorer

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Evaluate expansion model")
    parser.add_argument('--weights_file', required=True, help='Path to learned weights')
    parser.add_argument('--dataset', default='msmarco-passage/trec-dl-2019', help='Evaluation dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--semantic_model', default='all-MiniLM-L6-v2', help='Semantic model')
    parser.add_argument('--index_path', help='BM25 index path (optional)')
    parser.add_argument('--lucene_path', help='Lucene JAR path (optional)')
    parser.add_argument('--ablation_study', action='store_true', help='Run ablation study')

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    # Load learned weights
    alpha, beta, gamma = load_learned_weights(args.weights_file)
    logger.info(f"Loaded weights: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}")

    # Initialize components
    rm_expansion = RMExpansion()
    semantic_sim = SemanticSimilarity(args.semantic_model)

    bm25_scorer = None
    if args.index_path and BM25_AVAILABLE:
        if args.lucene_path:
            from src.utils.lucene_utils import initialize_lucene
            initialize_lucene(args.lucene_path)
        bm25_scorer = TokenBM25Scorer(args.index_path)
        logger.info("BM25 scorer initialized")

    # Load evaluation dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = ir_datasets.load(args.dataset)

    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    first_stage_runs = defaultdict(list)
    for scoreddoc in dataset.scoreddocs_iter():
        first_stage_runs[scoreddoc.query_id].append((scoreddoc.doc_id, scoreddoc.score))

    logger.info(f"Loaded {len(queries)} queries, {len(qrels)} qrels")

    # Initialize evaluator
    evaluator = TRECEvaluator(['ndcg_cut_10', 'ndcg_cut_100', 'map', 'recip_rank'])

    # Evaluate baseline (first-stage)
    baseline_results = evaluator.evaluate_run(dict(first_stage_runs), dict(qrels))
    logger.info("Baseline evaluation completed")

    # Evaluate with learned weights
    importance_weights_dict = compute_importance_weights(
        queries, dict(first_stage_runs), rm_expansion, semantic_sim, bm25_scorer,
        alpha=alpha, beta=beta, gamma=gamma
    )

    # For this example, we'll just use first-stage runs with importance scoring
    # In practice, you'd integrate with your multi-vector reranker
    our_method_results = dict(first_stage_runs)  # Placeholder
    our_results = evaluator.evaluate_run(our_method_results, dict(qrels))
    logger.info("Our method evaluation completed")

    # Compare results
    comparison = evaluator.compare_runs({
        'baseline': dict(first_stage_runs),
        'our_method': our_method_results
    }, dict(qrels), 'baseline')

    # Save results
    output_dir = ensure_dir(args.output_dir)
    results = {
        'dataset': args.dataset,
        'weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
        'baseline_metrics': baseline_results,
        'our_method_metrics': our_results,
        'comparison': comparison
    }

    save_json(results, output_dir / 'evaluation_results.json')

    # Print summary
    print("Evaluation Results:")
    print("=" * 50)
    for metric in ['ndcg_cut_10', 'map']:
        baseline = baseline_results[metric]
        ours = our_results[metric]
        improvement = ours - baseline
        print(f"{metric.upper():<12}: {baseline:.4f} → {ours:.4f} (+{improvement:.4f})")

    # Run ablation study if requested
    if args.ablation_study:
        logger.info("Running ablation study...")
        ablation_results = run_ablation_study(
            queries, dict(qrels), dict(first_stage_runs),
            rm_expansion, semantic_sim, bm25_scorer, evaluator
        )

        save_json(ablation_results, output_dir / 'ablation_results.json')

        print("\nAblation Study Results:")
        print("=" * 50)
        print(f"{'Method':<15} {'nDCG@10':<10} {'MAP':<10}")
        print("-" * 35)
        for method, scores in ablation_results.items():
            print(f"{method:<15} {scores['ndcg_cut_10']:<10.4f} {scores['map']:<10.4f}")


def compute_importance_weights(queries, first_stage_runs, rm_expansion,
                               semantic_sim, bm25_scorer, alpha, beta, gamma):
    """Compute importance weights for all queries."""
    # Implementation as shown above
    pass


def run_ablation_study(queries, qrels, first_stage_runs, rm_expansion,
                       semantic_sim, bm25_scorer, evaluator):
    """Run ablation study across different weight configurations."""
    # Implementation as shown above  
    pass


if __name__ == "__main__":
    main()
```

