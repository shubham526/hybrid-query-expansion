#!/usr/bin/env python3
"""
Test the evaluation function directly to see why L-BFGS-B isn't finding differences
"""

import logging
from pathlib import Path
import sys
from collections import defaultdict

# Add your project path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.core.bm25_scorer import TokenBM25Scorer
from src.models.memory_efficient_reranker import create_memory_efficient_reranker
from src.evaluation.evaluator import create_trec_dl_evaluator
from src.utils.lucene_utils import initialize_lucene
from src.utils.file_utils import load_json, load_trec_run
import ir_datasets


def test_evaluation_sensitivity():
    """Test if evaluation function is sensitive to weight changes"""

    print("🔍 DIAGNOSTIC: Testing Evaluation Function Sensitivity")
    print("=" * 60)

    # Load your exact setup
    index_path = "/home/user/hybrid-query-expansion/robust/index_all-MiniLM-L6-v2/disks45_nocr_trec-robust-2004_sentence-transformers_all-MiniLM-L6-v2/"
    lucene_path = "/home/user/lucene-10.1.0/modules/"
    feature_file = "/home/user/hybrid-query-expansion/robust/experiments/robust2004_in_domain/features/fold4/disks45_nocr_trec-robust-2004_fold_4_train_qids_features.json.gz"
    run_file = "/home/user/QDER/robust04/data/all/title.BM25_RM3_TUNED.run"
    query_ids_file = "/home/user/hybrid-query-expansion/robust/experiments/robust2004_in_domain/temp_fold_files/fold_4_train_qids.txt"

    # Initialize components
    print("1. Initializing components...")
    initialize_lucene(lucene_path)

    # Load data (subset for speed)
    print("2. Loading data...")
    features = load_json(feature_file)

    # Load validation queries (subset)
    with open(query_ids_file, 'r') as f:
        train_qids = {line.strip() for line in f if line.strip()}

    # Take first 5 queries for quick test
    sample_qids = list(train_qids)[:5]
    print(f"   Testing with {len(sample_qids)} sample queries")

    # Load dataset and runs
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
    all_queries = {q.query_id: q.title for q in dataset.queries_iter() if q.query_id in sample_qids}
    all_qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        if qrel.query_id in sample_qids:
            all_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    all_documents = {}
    for doc in dataset.docs_iter():
        all_documents[doc.doc_id] = doc.body
        if len(all_documents) > 10000:  # Limit for speed
            break

    first_stage_runs = load_trec_run(run_file)
    first_stage_runs = {qid: run for qid, run in first_stage_runs.items() if qid in sample_qids}

    print(f"   Loaded {len(all_queries)} queries, {len(all_documents)} docs")

    # Create reranker
    print("3. Creating reranker...")
    reranker = create_memory_efficient_reranker("all-MiniLM-L6-v2", large_candidate_sets=False)

    # Test different weight combinations
    print("4. Testing weight sensitivity...")
    test_weights = [
        (1.0, 1.0, 1.0),  # Equal weights
        (2.0, 1.0, 1.0),  # High RM
        (1.0, 2.0, 1.0),  # High BM25
        (1.0, 1.0, 2.0),  # High semantic
        (1.0, 0.0, 0.0),  # RM only
        (0.0, 1.0, 0.0),  # BM25 only
        (0.0, 0.0, 1.0),  # Semantic only
    ]

    results = {}

    for weights in test_weights:
        alpha, beta, gamma = weights
        print(f"   Testing weights α={alpha}, β={beta}, γ={gamma}...")

        try:
            # Simulate the evaluation function
            reranked_runs = {}

            for qid, query_text in all_queries.items():
                if qid not in features or qid not in first_stage_runs:
                    continue

                query_features = features[qid]

                # Compute importance weights using test weights
                importance_weights = {
                    term: (alpha * term_data['rm_weight'] +
                           beta * term_data['bm25_score'] +
                           gamma * term_data['semantic_score'])
                    for term, term_data in query_features['term_features'].items()
                }

                # Create expansion terms
                expansion_terms = [(term, term_data['rm_weight'])
                                   for term, term_data in query_features['term_features'].items()]

                # Prepare candidate results for this query
                candidate_results = []
                for doc_id, first_stage_score in first_stage_runs[qid][:100]:  # Top 100 for speed
                    doc_text = all_documents.get(doc_id, "")
                    if doc_text:
                        candidate_results.append((doc_id, doc_text, first_stage_score))

                if candidate_results:
                    try:
                        # Use streaming reranking
                        reranked_results = reranker.rerank_streaming(
                            query=query_text,
                            expansion_terms=expansion_terms,
                            importance_weights=importance_weights,
                            candidate_results=candidate_results,
                            top_k=100
                        )
                        reranked_runs[qid] = reranked_results
                    except Exception as e:
                        print(f"      Error reranking query {qid}: {e}")
                        reranked_runs[qid] = [(doc_id, score) for doc_id, _, score in candidate_results]
                else:
                    reranked_runs[qid] = []

            # Evaluate
            if reranked_runs:
                evaluator = create_trec_dl_evaluator()
                evaluation = evaluator.evaluate_run(reranked_runs, dict(all_qrels))
                score = evaluation.get("ndcg_cut_10", 0.0)
                results[weights] = score
                print(f"      → nDCG@10: {score:.4f}")
            else:
                results[weights] = 0.0
                print(f"      → No results")

        except Exception as e:
            print(f"      → Error: {e}")
            results[weights] = 0.0

    # Analyze results
    print("\n5. Analysis:")
    print("   Weight Combination → nDCG@10")
    print("   " + "-" * 35)

    score_variance = 0
    scores = list(results.values())
    if scores:
        mean_score = sum(scores) / len(scores)
        score_variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

    for weights, score in results.items():
        alpha, beta, gamma = weights
        print(f"   ({alpha:.1f}, {beta:.1f}, {gamma:.1f})      → {score:.4f}")

    print(f"\n   Score variance: {score_variance:.6f}")

    if score_variance < 0.0001:
        print("   ❌ PROBLEM: Scores are nearly identical!")
        print("   💡 This explains why L-BFGS-B converges immediately")
        print("   🔧 Issue is in the reranking pipeline")
    else:
        print("   ✅ Scores vary - evaluation function is sensitive")
        print("   💡 Problem might be in L-BFGS-B tolerance settings")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    results = test_evaluation_sensitivity()

    print("\n🎯 CONCLUSION:")
    if results:
        scores = list(results.values())
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        print(f"Score range: {min_score:.4f} - {max_score:.4f} (Δ = {score_range:.4f})")

        if score_range < 0.001:
            print("❌ Evaluation function is NOT sensitive to weight changes")
            print("🔧 Need to debug reranking pipeline")
        else:
            print("✅ Evaluation function IS sensitive to weight changes")
            print("🔧 Need to adjust L-BFGS-B settings (tolerance, bounds, etc.)")
    else:
        print("❌ Could not complete evaluation test")
        print("🔧 Check data loading and component initialization")