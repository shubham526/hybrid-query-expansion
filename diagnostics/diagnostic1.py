#!/usr/bin/env python3
"""
Quick diagnostic script to debug the optimization problem
"""

import logging
from pathlib import Path

# Add your project path
import sys

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.utils.lucene_utils import initialize_lucene
from src.utils.file_utils import load_json


def diagnose_expansion():
    """Check if RM expansion is working with the new Lucene backend"""

    # Initialize components
    index_path = "/home/user/hybrid-query-expansion/robust/index_all-MiniLM-L6-v2/disks45_nocr_trec-robust-2004_sentence-transformers_all-MiniLM-L6-v2/"
    lucene_path = "/home/user/lucene-10.1.0/modules/"

    print("🔍 DIAGNOSTIC: Testing RM Expansion")
    print("=" * 50)

    try:
        # Initialize Lucene
        print("1. Initializing Lucene...")
        initialize_lucene(lucene_path)
        print("   ✅ Lucene initialized")

        # Create RM expansion
        print("2. Creating RM expansion...")
        rm_expansion = RMExpansion(index_path)
        print("   ✅ RM expansion created")

        # Test expansion
        print("3. Testing expansion...")
        test_query = "machine learning algorithms"
        test_docs = [
            "Machine learning is a subset of artificial intelligence that uses statistical techniques.",
            "Neural networks are powerful algorithms for pattern recognition in data.",
            "Classification algorithms categorize data into different classes."
        ]
        test_scores = [0.9, 0.8, 0.7]

        expansion_terms = rm_expansion.expand_query(
            query=test_query,
            documents=test_docs,  # These should be ignored by Lucene
            scores=test_scores,  # These should be ignored by Lucene
            num_expansion_terms=10,
            rm_type="rm3"
        )

        print(f"   📊 Got {len(expansion_terms)} expansion terms:")
        for i, (term, weight) in enumerate(expansion_terms[:5], 1):
            print(f"      {i}. {term:<15} {weight:.4f}")

        if not expansion_terms:
            print("   ❌ NO EXPANSION TERMS! This is the problem!")
            return False
        else:
            print("   ✅ Expansion terms generated successfully")

    except Exception as e:
        print(f"   ❌ Error in RM expansion: {e}")
        return False

    return True


def diagnose_features():
    """Check if features are properly structured"""

    feature_file = "/home/user/hybrid-query-expansion/robust/experiments/robust2004_in_domain/features/fold4/disks45_nocr_trec-robust-2004_fold_4_train_qids_features.json.gz"

    print("\n🔍 DIAGNOSTIC: Testing Feature Loading")
    print("=" * 50)

    try:
        print("1. Loading features...")
        features = load_json(feature_file)
        print(f"   ✅ Loaded features for {len(features)} queries")

        # Check a sample query
        sample_qid = list(features.keys())[0]
        sample_features = features[sample_qid]

        print(f"2. Checking sample query {sample_qid}...")
        print(f"   Query text: {sample_features.get('query_text', 'MISSING')}")

        term_features = sample_features.get('term_features', {})
        print(f"   Number of expansion terms: {len(term_features)}")

        if term_features:
            sample_term = list(term_features.keys())[0]
            sample_term_data = term_features[sample_term]
            print(f"   Sample term '{sample_term}':")
            print(f"     RM weight: {sample_term_data.get('rm_weight', 'MISSING')}")
            print(f"     BM25 score: {sample_term_data.get('bm25_score', 'MISSING')}")
            print(f"     Semantic score: {sample_term_data.get('semantic_score', 'MISSING')}")

            # Check if scores vary
            rm_weights = [td.get('rm_weight', 0) for td in term_features.values()]
            bm25_scores = [td.get('bm25_score', 0) for td in term_features.values()]
            semantic_scores = [td.get('semantic_score', 0) for td in term_features.values()]

            print(f"   RM weight range: {min(rm_weights):.4f} - {max(rm_weights):.4f}")
            print(f"   BM25 score range: {min(bm25_scores):.4f} - {max(bm25_scores):.4f}")
            print(f"   Semantic score range: {min(semantic_scores):.4f} - {max(semantic_scores):.4f}")

            # Check for all-zero scores
            if all(score == 0 for score in bm25_scores):
                print("   ⚠️  WARNING: All BM25 scores are zero!")
            if all(score == 0 for score in semantic_scores):
                print("   ⚠️  WARNING: All semantic scores are zero!")

        else:
            print("   ❌ NO TERM FEATURES! This is a problem!")
            return False

    except Exception as e:
        print(f"   ❌ Error loading features: {e}")
        return False

    return True


def test_manual_weight_evaluation():
    """Manually test different weight combinations"""

    print("\n🔍 DIAGNOSTIC: Manual Weight Testing")
    print("=" * 50)

    # This would require loading your full evaluation pipeline
    # But you can add this if needed
    print("   TODO: Test (1,0,0), (0,1,0), (0,0,1) manually")
    print("   If all give same score, the evaluation function is broken")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("🚨 DEBUGGING OPTIMIZATION CONVERGENCE ISSUE")
    print("=" * 60)

    # Run diagnostics
    rm_ok = diagnose_expansion()
    features_ok = diagnose_features()

    print("\n📋 SUMMARY")
    print("=" * 30)
    print(f"RM Expansion: {'✅ OK' if rm_ok else '❌ BROKEN'}")
    print(f"Features: {'✅ OK' if features_ok else '❌ BROKEN'}")

    if not rm_ok:
        print("\n🔧 LIKELY CAUSE: RM expansion not working with Lucene")
        print("   - Check if index path is correct")
        print("   - Check if Lucene classes are loading properly")
        print("   - Verify documents are in the index")

    if not features_ok:
        print("\n🔧 LIKELY CAUSE: Features missing or malformed")
        print("   - Re-run create_training_data.py")
        print("   - Check BM25 scorer integration")

    print("\n💡 NEXT STEPS:")
    print("1. Run this diagnostic script")
    print("2. Fix any issues found")
    print("3. Re-run weight training")
    print("4. Should see iterations > 0 and weight changes")