#!/usr/bin/env python3
"""
Complete diagnostic script to identify why L-BFGS-B optimization is failing.
Tests all components: RM expansion, BM25 scoring, feature loading, and evaluation function.
"""

import logging
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

# Import your project's modules
from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.models.memory_efficient_reranker import create_memory_efficient_reranker
from src.evaluation.evaluator import create_trec_dl_evaluator
from src.utils.file_utils import load_json, load_trec_run
from src.utils.lucene_utils import initialize_lucene

# Conditionally import BM25
try:
    from src.core.bm25_scorer import TokenBM25Scorer

    BM25_AVAILABLE = True
except ImportError:
    TokenBM25Scorer = None
    BM25_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - UPDATE THESE PATHS FOR YOUR SETUP
CONFIG = {
    'index_path': "/home/user/hybrid-query-expansion/robust/index_all-MiniLM-L6-v2/disks45_nocr_trec-robust-2004_sentence-transformers_all-MiniLM-L6-v2/",
    'lucene_path': "/home/user/lucene-10.1.0/modules/",
    'feature_file': "/home/user/hybrid-query-expansion/robust/experiments/robust2004_in_domain/features/fold4/disks45_nocr_trec-robust-2004_fold_4_train_qids_features.json.gz",
    'run_file': "/home/user/QDER/robust04/data/all/title.BM25_RM3_TUNED.run",
    'query_ids_file': "/home/user/hybrid-query-expansion/robust/experiments/robust2004_in_domain/temp_fold_files/fold_4_train_qids.txt",
    'dataset_name': "disks45/nocr/trec-robust-2004",
    'semantic_model': "all-MiniLM-L6-v2"
}


def test_1_rm_expansion():
    """Test 1: RM Expansion functionality"""
    logger.info("=" * 60)
    logger.info("TEST 1: RM EXPANSION")
    logger.info("=" * 60)

    try:
        # Initialize Lucene
        logger.info("Initializing Lucene...")
        success = initialize_lucene(CONFIG['lucene_path'])
        if not success:
            logger.error("❌ Lucene initialization failed")
            return False
        logger.info("✅ Lucene initialized")

        # Create RM expansion
        logger.info("Creating RM expansion...")
        rm_expansion = RMExpansion(CONFIG['index_path'])
        logger.info("✅ RM expansion created")

        # Test expansion
        logger.info("Testing RM expansion...")
        test_query = "machine learning algorithms"
        test_docs = [
            "Machine learning is a subset of artificial intelligence that uses statistical techniques.",
            "Neural networks are powerful algorithms for pattern recognition in data.",
            "Classification algorithms categorize data into different classes."
        ]
        test_scores = [0.9, 0.8, 0.7]

        expansion_terms = rm_expansion.expand_query(
            query=test_query,
            documents=test_docs,
            scores=test_scores,
            num_expansion_terms=10,
            rm_type="rm3"
        )

        logger.info(f"📊 Generated {len(expansion_terms)} expansion terms:")
        for i, (term, weight) in enumerate(expansion_terms[:5], 1):
            logger.info(f"  {i}. {term:<15} {weight:.4f}")

        if not expansion_terms:
            logger.error("❌ NO EXPANSION TERMS generated!")
            return False
        else:
            logger.info("✅ RM expansion working correctly")
            return True

    except Exception as e:
        logger.error(f"❌ RM expansion test failed: {e}")
        return False


def test_2_feature_loading():
    """Test 2: Feature loading and structure"""
    logger.info("=" * 60)
    logger.info("TEST 2: FEATURE LOADING")
    logger.info("=" * 60)

    try:
        logger.info("Loading features...")
        features = load_json(CONFIG['feature_file'])
        logger.info(f"✅ Loaded features for {len(features)} queries")

        # Check sample query
        sample_qid = list(features.keys())[0]
        sample_features = features[sample_qid]

        logger.info(f"Checking sample query {sample_qid}...")
        logger.info(f"  Query text: {sample_features.get('query_text', 'MISSING')[:100]}...")

        term_features = sample_features.get('term_features', {})
        logger.info(f"  Number of expansion terms: {len(term_features)}")

        if term_features:
            sample_term = list(term_features.keys())[0]
            sample_term_data = term_features[sample_term]
            logger.info(f"  Sample term '{sample_term}':")
            logger.info(f"    RM weight: {sample_term_data.get('rm_weight', 'MISSING')}")
            logger.info(f"    BM25 score: {sample_term_data.get('bm25_score', 'MISSING')}")
            logger.info(f"    Semantic score: {sample_term_data.get('semantic_score', 'MISSING')}")

            # Check score ranges
            rm_weights = [td.get('rm_weight', 0) for td in term_features.values()]
            bm25_scores = [td.get('bm25_score', 0) for td in term_features.values()]
            semantic_scores = [td.get('semantic_score', 0) for td in term_features.values()]

            logger.info(f"  Score ranges:")
            logger.info(f"    RM weights: {min(rm_weights):.4f} - {max(rm_weights):.4f}")
            logger.info(f"    BM25 scores: {min(bm25_scores):.4f} - {max(bm25_scores):.4f}")
            logger.info(f"    Semantic scores: {min(semantic_scores):.4f} - {max(semantic_scores):.4f}")

            # Check for all-zero scores
            if all(score == 0 for score in bm25_scores):
                logger.warning("⚠️ WARNING: All BM25 scores are zero!")
            if all(score == 0 for score in semantic_scores):
                logger.warning("⚠️ WARNING: All semantic scores are zero!")

            logger.info("✅ Features loaded and structured correctly")
            return features
        else:
            logger.error("❌ NO TERM FEATURES found!")
            return None

    except Exception as e:
        logger.error(f"❌ Feature loading failed: {e}")
        return None


def test_3_bm25_diagnostics(features):
    """Test 3: BM25 scorer diagnostics"""
    logger.info("=" * 60)
    logger.info("TEST 3: BM25 DIAGNOSTICS")
    logger.info("=" * 60)

    if not BM25_AVAILABLE:
        logger.warning("❌ BM25 not available - skipping test")
        return None

    try:
        # Create BM25 scorer
        bm25_scorer = TokenBM25Scorer(CONFIG['index_path'])
        logger.info("✅ BM25 scorer created")

        # Load run file to get document IDs
        runs = load_trec_run(CONFIG['run_file'])
        logger.info(f"✅ Loaded runs for {len(runs)} queries")

        # Test 3.1: Document existence
        logger.info("\nTest 3.1: Document existence check")
        sample_doc_ids = []
        for query_id, docs in list(runs.items())[:3]:
            sample_doc_ids.extend([doc_id for doc_id, _ in docs[:3]])

        found_any = False
        for doc_id in sample_doc_ids[:5]:
            try:
                common_word_scores = bm25_scorer.compute_bm25_term_weight(doc_id, ["the", "and", "a"])
                if any(score > 0 for score in common_word_scores.values()):
                    logger.info(f"  ✓ Document {doc_id} found! Scores: {common_word_scores}")
                    found_any = True
                else:
                    logger.warning(f"  ⚠️ Document {doc_id} exists but no scores for common words")
            except Exception as e:
                logger.error(f"  ✗ Error with document {doc_id}: {e}")

        if not found_any:
            logger.error("🚨 CRITICAL: No documents found in index!")
            return None

        # Test 3.2: Expansion term coverage
        logger.info("\nTest 3.2: Expansion term coverage in reference documents")

        expansion_term_coverage = []
        for query_id, query_data in list(features.items())[:3]:
            reference_doc_id = query_data.get('reference_doc_id')
            expansion_terms = list(query_data['term_features'].keys())

            if not reference_doc_id or not expansion_terms:
                continue

            logger.info(f"\n  Query {query_id}:")
            logger.info(f"    Reference doc: {reference_doc_id}")
            logger.info(f"    Expansion terms: {expansion_terms[:5]}...")

            try:
                bm25_scores = bm25_scorer.compute_bm25_term_weight(reference_doc_id, expansion_terms)

                zero_count = sum(1 for score in bm25_scores.values() if score == 0.0)
                nonzero_count = len(bm25_scores) - zero_count
                coverage_ratio = nonzero_count / len(bm25_scores) if bm25_scores else 0
                expansion_term_coverage.append(coverage_ratio)

                logger.info(f"    BM25 coverage: {nonzero_count}/{len(bm25_scores)} terms ({coverage_ratio:.1%})")

                if nonzero_count > 0:
                    nonzero_scores = {term: score for term, score in bm25_scores.items() if score > 0}
                    logger.info(f"    Sample non-zero scores: {dict(list(nonzero_scores.items())[:3])}")
                else:
                    logger.warning(f"    ⚠️ ALL expansion terms have zero BM25 scores!")

            except Exception as e:
                logger.error(f"    Error computing BM25 for query {query_id}: {e}")

        # Summary
        if expansion_term_coverage:
            avg_coverage = sum(expansion_term_coverage) / len(expansion_term_coverage)
            logger.info(f"\nBM25 SUMMARY:")
            logger.info(f"  Average expansion term coverage: {avg_coverage:.1%}")

            if avg_coverage < 0.1:
                logger.error("🚨 ISSUE CONFIRMED: Very low expansion term coverage!")
                logger.error("   → This explains why BM25 component contributes nothing")
                return None
            elif avg_coverage < 0.3:
                logger.warning("⚠️ ISSUE LIKELY: Low expansion term coverage")
                return bm25_scorer
            else:
                logger.info("✅ Decent expansion term coverage")
                return bm25_scorer

        return bm25_scorer

    except Exception as e:
        logger.error(f"❌ BM25 test failed: {e}")
        return None


def test_3_updated_bm25_diagnostics(features):
    """Test 3: Updated BM25 scorer diagnostics - tests BOTH old and new methods"""
    logger.info("=" * 60)
    logger.info("TEST 3: UPDATED BM25 DIAGNOSTICS")
    logger.info("=" * 60)

    if not BM25_AVAILABLE:
        logger.warning("❌ BM25 not available - skipping test")
        return None

    try:
        # Create BM25 scorer
        bm25_scorer = TokenBM25Scorer(CONFIG['index_path'])
        logger.info("✅ BM25 scorer created")

        # Load run file to get document IDs
        runs = load_trec_run(CONFIG['run_file'])
        logger.info(f"✅ Loaded runs for {len(runs)} queries")

        # Test both old and new methods
        logger.info("\n🔍 TESTING BOTH OLD AND NEW BM25 METHODS:")

        # Get sample data
        sample_query_data = list(features.values())[0]
        sample_query_id = list(features.keys())[0]
        reference_doc_id = sample_query_data.get('reference_doc_id')
        expansion_terms = list(sample_query_data['term_features'].keys())[:5]  # Test first 5 terms

        logger.info(f"Sample query: {sample_query_id}")
        logger.info(f"Reference doc: {reference_doc_id}")
        logger.info(f"Testing terms: {expansion_terms}")

        # TEST 1: Old document-level method (should fail)
        logger.info("\n--- Testing OLD document-level method ---")
        if reference_doc_id:
            try:
                old_scores = bm25_scorer.compute_bm25_term_weight(reference_doc_id, expansion_terms)
                old_nonzero = sum(1 for score in old_scores.values() if score > 0)
                logger.info(f"Old method scores: {old_scores}")
                logger.info(f"Non-zero scores: {old_nonzero}/{len(expansion_terms)}")

                if old_nonzero == 0:
                    logger.info("❌ OLD METHOD: All scores are zero (as expected)")
                else:
                    logger.info("✅ OLD METHOD: Some non-zero scores (unexpected!)")

            except Exception as e:
                logger.error(f"Old method error: {e}")
                old_scores = {}
                old_nonzero = 0
        else:
            logger.warning("No reference doc ID available for old method test")
            old_scores = {}
            old_nonzero = 0

        # TEST 2: New collection-level method (should work)
        logger.info("\n--- Testing NEW collection-level method ---")
        if hasattr(bm25_scorer, 'compute_collection_level_bm25'):
            try:
                new_scores = bm25_scorer.compute_collection_level_bm25(expansion_terms)
                new_nonzero = sum(1 for score in new_scores.values() if score > 0)
                logger.info(f"New method scores: {new_scores}")
                logger.info(f"Non-zero scores: {new_nonzero}/{len(expansion_terms)}")

                if new_nonzero > 0:
                    logger.info("✅ NEW METHOD: Collection-level BM25 is working!")

                    # Show score comparison
                    logger.info(f"\n📊 COMPARISON:")
                    logger.info(f"Old method non-zero: {old_nonzero}/{len(expansion_terms)}")
                    logger.info(f"New method non-zero: {new_nonzero}/{len(expansion_terms)}")
                    logger.info(f"Improvement: +{new_nonzero - old_nonzero} terms with scores")

                    return bm25_scorer
                else:
                    logger.error("❌ NEW METHOD: Still getting all zero scores!")
                    logger.error("   → Collection-level method not working as expected")
                    return None

            except Exception as e:
                logger.error(f"New method error: {e}")
                logger.error("   → Collection-level method may not be implemented correctly")
                return None
        else:
            logger.error("❌ COLLECTION-LEVEL METHOD NOT FOUND!")
            logger.error("   → compute_collection_level_bm25 method not implemented")
            logger.error("   → Need to add this method to TokenBM25Scorer class")
            return None

        # TEST 3: Test with common words (should definitely work)
        logger.info("\n--- Testing with common words ---")
        common_words = ["the", "and", "of", "to", "a"]

        if hasattr(bm25_scorer, 'compute_collection_level_bm25'):
            try:
                common_scores = bm25_scorer.compute_collection_level_bm25(common_words)
                common_nonzero = sum(1 for score in common_scores.values() if score > 0)
                logger.info(f"Common words scores: {common_scores}")
                logger.info(f"Non-zero scores: {common_nonzero}/{len(common_words)}")

                if common_nonzero > 0:
                    logger.info("✅ Collection-level BM25 works for common words")
                else:
                    logger.error("❌ Even common words get zero scores - deep indexing issue")

            except Exception as e:
                logger.error(f"Common words test error: {e}")

        # TEST 4: Test multiple queries
        logger.info("\n--- Testing multiple queries ---")
        if hasattr(bm25_scorer, 'compute_collection_level_bm25'):
            total_terms_tested = 0
            total_nonzero_terms = 0

            for query_id, query_data in list(features.items())[:3]:  # Test 3 queries
                terms = list(query_data['term_features'].keys())[:3]  # Test 3 terms each

                try:
                    scores = bm25_scorer.compute_collection_level_bm25(terms)
                    nonzero = sum(1 for score in scores.values() if score > 0)

                    total_terms_tested += len(terms)
                    total_nonzero_terms += nonzero

                    logger.info(f"Query {query_id}: {nonzero}/{len(terms)} terms with scores")

                except Exception as e:
                    logger.warning(f"Error testing query {query_id}: {e}")

            if total_terms_tested > 0:
                success_rate = total_nonzero_terms / total_terms_tested
                logger.info(
                    f"\n📈 OVERALL SUCCESS RATE: {total_nonzero_terms}/{total_terms_tested} ({success_rate:.1%})")

                if success_rate > 0.5:  # More than 50% of terms get scores
                    logger.info("✅ Collection-level BM25 is working well!")
                    return bm25_scorer
                elif success_rate > 0.1:  # More than 10% of terms get scores
                    logger.warning("⚠️ Collection-level BM25 partially working")
                    return bm25_scorer
                else:
                    logger.error("❌ Collection-level BM25 success rate too low")
                    return None

        return None

    except Exception as e:
        logger.error(f"❌ BM25 diagnostic failed: {e}")
        return None


def quick_implementation_check():
    """Quick check to verify the collection-level method is properly implemented"""
    logger.info("=" * 60)
    logger.info("QUICK IMPLEMENTATION CHECK")
    logger.info("=" * 60)

    try:
        # Initialize BM25 scorer
        initialize_lucene(CONFIG['lucene_path'])
        bm25_scorer = TokenBM25Scorer(CONFIG['index_path'])

        # Check if method exists
        if hasattr(bm25_scorer, 'compute_collection_level_bm25'):
            logger.info("✅ compute_collection_level_bm25 method found!")

            # Quick test
            test_terms = ["machine", "learning"]
            logger.info(f"Testing with terms: {test_terms}")

            try:
                scores = bm25_scorer.compute_collection_level_bm25(test_terms)
                logger.info(f"Result: {scores}")

                if any(score > 0 for score in scores.values()):
                    logger.info("✅ Collection-level BM25 implementation working!")
                    return True
                else:
                    logger.warning("⚠️ Method exists but returns zero scores")
                    return False

            except Exception as e:
                logger.error(f"❌ Method exists but throws error: {e}")
                return False
        else:
            logger.error("❌ compute_collection_level_bm25 method NOT FOUND!")
            logger.error("   You need to add this method to src/core/bm25_scorer.py")
            return False

    except Exception as e:
        logger.error(f"❌ Implementation check failed: {e}")
        return False


def test_4_evaluation_sensitivity(features, bm25_scorer):
    """Test 4: Evaluation function sensitivity"""
    logger.info("=" * 60)
    logger.info("TEST 4: EVALUATION FUNCTION SENSITIVITY")
    logger.info("=" * 60)

    try:
        # Load data for evaluation
        logger.info("Loading evaluation data...")

        # Load training query IDs
        with open(CONFIG['query_ids_file'], 'r') as f:
            train_qids = {line.strip() for line in f if line.strip()}

        # Take first 3 queries for quick test
        sample_qids = list(train_qids)[:3]
        logger.info(f"Testing with {len(sample_qids)} sample queries")

        # Load dataset components
        dataset = ir_datasets.load(CONFIG['dataset_name'])
        all_queries = {q.query_id: q.title for q in dataset.queries_iter() if q.query_id in sample_qids}

        all_qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            if qrel.query_id in sample_qids:
                all_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        # Load subset of documents (for memory)
        all_documents = {}
        doc_count = 0
        for doc in dataset.docs_iter():
            all_documents[doc.doc_id] = doc.text
            doc_count += 1
            if doc_count > 20000:  # Limit for speed
                break

        # Load runs
        first_stage_runs = load_trec_run(CONFIG['run_file'])
        first_stage_runs = {qid: run for qid, run in first_stage_runs.items() if qid in sample_qids}

        logger.info(f"✅ Loaded {len(all_queries)} queries, {len(all_documents)} docs")

        # Create reranker
        logger.info("Creating reranker...")
        reranker = create_memory_efficient_reranker(CONFIG['semantic_model'], large_candidate_sets=False)

        # Test different weight combinations
        logger.info("Testing weight sensitivity...")
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
            logger.info(f"  Testing weights α={alpha}, β={beta}, γ={gamma}...")

            try:
                reranked_runs = {}

                for qid, query_text in all_queries.items():
                    if qid not in features or qid not in first_stage_runs:
                        continue

                    query_features = features[qid]

                    # Compute importance weights
                    importance_weights = {
                        term: (alpha * term_data['rm_weight'] +
                               beta * term_data['bm25_score'] +
                               gamma * term_data['semantic_score'])
                        for term, term_data in query_features['term_features'].items()
                    }

                    # Create expansion terms
                    expansion_terms = [(term, term_data['rm_weight'])
                                       for term, term_data in query_features['term_features'].items()]

                    # Prepare candidate results (top 50 for speed)
                    candidate_results = []
                    for doc_id, first_stage_score in first_stage_runs[qid][:50]:
                        doc_text = all_documents.get(doc_id, "")
                        if doc_text:
                            candidate_results.append((doc_id, doc_text, first_stage_score))

                    if candidate_results:
                        try:
                            reranked_results = reranker.rerank_streaming(
                                query=query_text,
                                expansion_terms=expansion_terms,
                                importance_weights=importance_weights,
                                candidate_results=candidate_results,
                                top_k=50
                            )
                            reranked_runs[qid] = reranked_results
                        except Exception as e:
                            logger.warning(f"      Error reranking query {qid}: {e}")
                            reranked_runs[qid] = [(doc_id, score) for doc_id, _, score in candidate_results]
                    else:
                        reranked_runs[qid] = []

                # Evaluate
                if reranked_runs:
                    evaluator = create_trec_dl_evaluator()
                    evaluation = evaluator.evaluate_run(reranked_runs, dict(all_qrels))
                    score = evaluation.get("ndcg_cut_10", 0.0)
                    results[weights] = score
                    logger.info(f"      → nDCG@10: {score:.4f}")
                else:
                    results[weights] = 0.0
                    logger.info(f"      → No results")

            except Exception as e:
                logger.error(f"      → Error: {e}")
                results[weights] = 0.0

        # Analyze results
        logger.info("\nEVALUATION SENSITIVITY ANALYSIS:")
        logger.info("  Weight Combination → nDCG@10")
        logger.info("  " + "-" * 35)

        scores = list(results.values())
        if scores:
            mean_score = sum(scores) / len(scores)
            score_variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
        else:
            score_variance = 0
            score_range = 0

        for weights, score in results.items():
            alpha, beta, gamma = weights
            logger.info(f"  ({alpha:.1f}, {beta:.1f}, {gamma:.1f})      → {score:.4f}")

        logger.info(f"\n  Score variance: {score_variance:.6f}")
        logger.info(f"  Score range: {score_range:.4f}")

        if score_variance < 0.0001:
            logger.error("🚨 PROBLEM: Scores are nearly identical!")
            logger.error("   → This explains why L-BFGS-B converges immediately")
            logger.error("   → Issue is in the reranking pipeline")
            return False
        elif score_range < 0.001:
            logger.warning("⚠️ PROBLEM: Very small score differences")
            logger.warning("   → L-BFGS-B tolerance may be too strict")
            return True
        else:
            logger.info("✅ Scores vary significantly - evaluation function is sensitive")
            logger.info("   → Problem is likely in L-BFGS-B settings")
            return True

    except Exception as e:
        logger.error(f"❌ Evaluation sensitivity test failed: {e}")
        return False


def test_5_run_format_check():
    """Test 5: Check baseline run file format"""
    logger.info("=" * 60)
    logger.info("TEST 5: BASELINE RUN FORMAT CHECK")
    logger.info("=" * 60)

    try:
        logger.info(f"Checking run file: {CONFIG['run_file']}")

        with open(CONFIG['run_file'], 'r') as f:
            lines = f.readlines()

        logger.info(f"Total lines in run file: {len(lines)}")

        # Show first few lines
        logger.info("First 5 lines:")
        for i, line in enumerate(lines[:5]):
            parts = line.strip().split()
            logger.info(f"  Line {i}: {parts}")
            if len(parts) >= 6:
                logger.info(f"    Query ID: '{parts[0]}', Doc ID: '{parts[2]}', Score: {parts[4]}")

        # Check document ID patterns
        sample_doc_ids = []
        for line in lines[:20]:
            parts = line.strip().split()
            if len(parts) >= 6:
                sample_doc_ids.append(parts[2])

        logger.info(f"Sample document IDs: {sample_doc_ids[:10]}")

        # Analyze ID patterns
        robust_pattern_count = sum(1 for doc_id in sample_doc_ids if any(c.isalpha() for c in doc_id))
        numeric_pattern_count = sum(1 for doc_id in sample_doc_ids if doc_id.isdigit())

        logger.info(f"ID format analysis:")
        logger.info(f"  Alphanumeric (Robust-like): {robust_pattern_count}")
        logger.info(f"  Numeric only: {numeric_pattern_count}")

        logger.info("✅ Run file format check completed")
        return True

    except Exception as e:
        logger.error(f"❌ Run format check failed: {e}")
        return False


def main():
    """Run all diagnostic tests"""
    logger.info("🚨 COMPREHENSIVE OPTIMIZATION DIAGNOSTIC")
    logger.info("=" * 80)
    logger.info("Testing all components to identify why L-BFGS-B optimization fails")
    logger.info("=" * 80)

    # Run implementation check first
    impl_ok = quick_implementation_check()

    if not impl_ok:
        logger.error("🚨 STOP: Collection-level BM25 not properly implemented!")
        logger.error("Add the compute_collection_level_bm25 method to TokenBM25Scorer first")
        return

    # Test 1: RM Expansion
    rm_ok = test_1_rm_expansion()

    # Test 2: Feature Loading
    features = test_2_feature_loading()
    features_ok = features is not None

    # Test 3: BM25 Diagnostics
    bm25_scorer = test_3_updated_bm25_diagnostics(features) if features else None
    bm25_ok = bm25_scorer is not None

    # Test 4: Evaluation Sensitivity (if features available)
    eval_ok = test_4_evaluation_sensitivity(features, bm25_scorer) if features else False

    # Test 5: Run Format Check
    run_ok = test_5_run_format_check()

    # Final Summary
    logger.info("=" * 80)
    logger.info("FINAL DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)

    logger.info(f"RM Expansion:        {'✅ OK' if rm_ok else '❌ BROKEN'}")
    logger.info(f"Feature Loading:     {'✅ OK' if features_ok else '❌ BROKEN'}")
    logger.info(f"BM25 Scorer:         {'✅ OK' if bm25_ok else '❌ BROKEN'}")
    logger.info(f"Evaluation Function: {'✅ OK' if eval_ok else '❌ BROKEN'}")
    logger.info(f"Run File Format:     {'✅ OK' if run_ok else '❌ BROKEN'}")

    # Diagnosis
    logger.info("\n🔧 DIAGNOSIS & RECOMMENDATIONS:")

    if not bm25_ok:
        logger.info("🚨 PRIMARY ISSUE: BM25 scoring is broken")
        logger.info("   → Expansion terms not found in reference documents")
        logger.info("   → All BM25 scores are zero, β component contributes nothing")
        logger.info("   → This creates a flat optimization landscape")
        logger.info("\n💡 SOLUTIONS:")
        logger.info("   1. Use collection-level BM25 instead of document-level")
        logger.info("   2. Aggregate BM25 across all PRF documents")
        logger.info("   3. Run with RM + Semantic only (set β=0)")

    if not eval_ok and features_ok:
        logger.info("🚨 SECONDARY ISSUE: Evaluation function not sensitive")
        logger.info("   → Different weight combinations yield same scores")
        logger.info("   → Check reranking pipeline")
        logger.info("\n💡 SOLUTIONS:")
        logger.info("   1. Debug multi-vector reranking")
        logger.info("   2. Adjust L-BFGS-B tolerance settings")
        logger.info("   3. Use grid search instead of L-BFGS-B")

    if rm_ok and features_ok and bm25_ok and eval_ok:
        logger.info("✅ ALL COMPONENTS WORKING")
        logger.info("   → Issue is likely in L-BFGS-B settings")
        logger.info("\n💡 SOLUTIONS:")
        logger.info("   1. Increase maxiter (50 → 200)")
        logger.info("   2. Loosen tolerance (1e-4 → 1e-6)")
        logger.info("   3. Adjust weight bounds")
        logger.info("   4. Try different optimizer (grid search)")

    logger.info("\n🎯 NEXT STEPS:")
    logger.info("1. Fix the primary issues identified above")
    logger.info("2. Re-run weight training")
    logger.info("3. Should see iterations > 0 and weight changes")
    logger.info("4. Monitor convergence and score improvements")


if __name__ == "__main__":
    main()