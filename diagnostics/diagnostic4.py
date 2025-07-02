#!/usr/bin/env python3
"""
Debug the collection-level BM25 implementation to find why it returns zero scores
"""


# DEBUGGING STEPS FOR COLLECTION-LEVEL BM25
# ==========================================

def debug_collection_bm25_step_by_step():
    """
    Step-by-step debugging of why collection-level BM25 returns zeros
    """

    print("🔍 DEBUGGING COLLECTION-LEVEL BM25")
    print("=" * 50)

    # Initialize
    from src.core.bm25_scorer import TokenBM25Scorer
    from src.utils.lucene_utils import initialize_lucene

    initialize_lucene("/home/user/lucene-10.1.0/modules/")
    bm25_scorer = TokenBM25Scorer(
        "/home/user/hybrid-query-expansion/robust/index_all-MiniLM-L6-v2/disks45_nocr_trec-robust-2004_sentence-transformers_all-MiniLM-L6-v2/")

    test_term = "machine"
    print(f"Testing with term: '{test_term}'")

    # STEP 1: Check if we can create the term query
    print("\nSTEP 1: Creating term query...")
    try:
        term_query = bm25_scorer.TermQuery(bm25_scorer.Term("contents", test_term))
        print(f"✅ Term query created: {term_query}")
    except Exception as e:
        print(f"❌ Error creating term query: {e}")
        return

    # STEP 2: Check if search returns any hits
    print("\nSTEP 2: Searching for term...")
    try:
        hits = bm25_scorer.searcher.search(term_query, 100)
        total_hits = hits.totalHits.value()
        print(f"📊 Total hits for '{test_term}': {total_hits}")

        if total_hits == 0:
            print(f"❌ PROBLEM: No documents contain '{test_term}'")
            print("   This could mean:")
            print("   1. Term doesn't exist in index")
            print("   2. Term is being analyzed differently")
            print("   3. Index field name is wrong")
            return
        else:
            print(f"✅ Found {total_hits} documents containing '{test_term}'")

    except Exception as e:
        print(f"❌ Error during search: {e}")
        return

    # STEP 3: Check the scores of returned documents
    print("\nSTEP 3: Examining document scores...")
    try:
        if len(hits.scoreDocs) > 0:
            print(f"📈 Score details for top documents:")
            for i, score_doc in enumerate(hits.scoreDocs[:5]):
                print(f"  Doc {i + 1}: score = {score_doc.score}")

            # Calculate what our method should return
            top_scores = [hit.score for hit in hits.scoreDocs[:10]]
            avg_score = sum(top_scores) / len(top_scores)
            print(f"📊 Top 10 average score: {avg_score}")

            if avg_score == 0.0:
                print("❌ PROBLEM: All document scores are zero!")
                print("   This means BM25 scoring itself is broken")
            else:
                print(f"✅ Expected result should be: {avg_score}")

        else:
            print("❌ No score documents returned")

    except Exception as e:
        print(f"❌ Error examining scores: {e}")
        return

    # STEP 4: Test the actual method implementation
    print("\nSTEP 4: Testing actual method...")
    try:
        result = bm25_scorer.compute_collection_level_bm25([test_term])
        print(f"🎯 Method result: {result}")

        if result[test_term] == 0.0:
            print("❌ Method returns zero despite having hits with scores!")
            print("   → There's a bug in the implementation")
        else:
            print("✅ Method working correctly")

    except Exception as e:
        print(f"❌ Error in method: {e}")

    # STEP 5: Test with different terms
    print("\nSTEP 5: Testing with multiple terms...")
    test_terms = ["the", "and", "machine", "learning", "algorithm", "document", "text"]

    for term in test_terms:
        try:
            term_query = bm25_scorer.TermQuery(bm25_scorer.Term("contents", term))
            hits = bm25_scorer.searcher.search(term_query, 10)
            hit_count = hits.totalHits.value()

            if hit_count > 0:
                avg_score = sum(hit.score for hit in hits.scoreDocs) / len(hits.scoreDocs)
                print(f"  '{term}': {hit_count} hits, avg score = {avg_score:.4f}")
            else:
                print(f"  '{term}': 0 hits")

        except Exception as e:
            print(f"  '{term}': ERROR - {e}")


# COMMON ISSUES AND FIXES
# =======================

def check_common_issues():
    """Check for common issues that cause zero scores"""

    print("\n🔧 CHECKING COMMON ISSUES")
    print("=" * 30)

    from src.core.bm25_scorer import TokenBM25Scorer
    from src.utils.lucene_utils import initialize_lucene

    initialize_lucene("/home/user/lucene-10.1.0/modules/")
    bm25_scorer = TokenBM25Scorer(
        "/home/user/hybrid-query-expansion/robust/index_all-MiniLM-L6-v2/disks45_nocr_trec-robust-2004_sentence-transformers_all-MiniLM-L6-v2/")

    # ISSUE 1: Check if similarity is set correctly
    print("1. Checking similarity setting...")
    try:
        similarity = bm25_scorer.searcher.getSimilarity()
        print(f"   Current similarity: {similarity}")

        # Make sure it's BM25
        if "BM25" in str(similarity):
            print("   ✅ BM25 similarity is set")
        else:
            print("   ⚠️ Not using BM25 similarity")

    except Exception as e:
        print(f"   ❌ Error checking similarity: {e}")

    # ISSUE 2: Check index statistics
    print("\n2. Checking index statistics...")
    try:
        reader = bm25_scorer.searcher.getIndexReader()
        num_docs = reader.numDocs()
        print(f"   Total documents in index: {num_docs}")

        if num_docs == 0:
            print("   ❌ CRITICAL: Index is empty!")
            return
        else:
            print("   ✅ Index has documents")

    except Exception as e:
        print(f"   ❌ Error checking index: {e}")

    # ISSUE 3: Check field name
    print("\n3. Checking field name...")
    try:
        # Try to get a sample document
        reader = bm25_scorer.searcher.getIndexReader()
        if reader.numDocs() > 0:
            doc = reader.document(0)
            fields = [field.name() for field in doc.getFields()]
            print(f"   Available fields: {fields}")

            if "contents" in fields:
                print("   ✅ 'contents' field exists")
                content = doc.get("contents")
                if content:
                    print(f"   Sample content: '{content[:100]}...'")
                else:
                    print("   ⚠️ 'contents' field is empty")
            else:
                print("   ❌ 'contents' field missing!")
                print("   Available fields:", fields)

    except Exception as e:
        print(f"   ❌ Error checking fields: {e}")

    # ISSUE 4: Check analyzer consistency
    print("\n4. Testing analyzer...")
    try:
        # Test how analyzer processes terms
        analyzer = bm25_scorer.analyzer if hasattr(bm25_scorer, 'analyzer') else None

        if analyzer:
            test_text = "machine learning algorithms"
            print(f"   Testing analyzer with: '{test_text}'")

            # This is tricky without direct access to analyzer results
            # But we can test if terms match
            print("   ✅ Analyzer available (detailed testing needed)")
        else:
            print("   ⚠️ Analyzer not accessible")

    except Exception as e:
        print(f"   ❌ Error testing analyzer: {e}")


# CORRECTED IMPLEMENTATION
# ========================

def get_corrected_implementation():
    """
    Show a corrected implementation that handles common issues
    """

    print("\n🔧 CORRECTED IMPLEMENTATION")
    print("=" * 30)

    corrected_code = '''
def compute_collection_level_bm25(self, terms: List[str], max_docs: int = 100) -> Dict[str, float]:
    """
    CORRECTED: Score terms against entire collection with better error handling
    """
    term_scores = {}

    for term in terms:
        try:
            # STEP 1: Create term (with debugging)
            term_obj = self.Term("contents", term)
            term_query = self.TermQuery(term_obj)

            # STEP 2: Search with debugging
            hits = self.searcher.search(term_query, max_docs)
            total_hits = hits.totalHits.value()

            # DEBUG: Log what we find
            logger.debug(f"Term '{term}': {total_hits} hits")

            if total_hits > 0:
                # STEP 3: Get scores with better error handling
                all_scores = []
                for hit in hits.scoreDocs:
                    if hit.score > 0:  # Only include positive scores
                        all_scores.append(hit.score)

                if all_scores:
                    # Use top-K average
                    top_k = min(10, len(all_scores))
                    avg_score = sum(all_scores[:top_k]) / top_k
                    term_scores[term] = float(avg_score)  # Ensure it's a Python float

                    # DEBUG: Log the result
                    logger.debug(f"Term '{term}': avg_score = {avg_score:.4f}")
                else:
                    # All scores were zero
                    term_scores[term] = 0.0
                    logger.debug(f"Term '{term}': all scores were zero")
            else:
                # No hits found
                term_scores[term] = 0.0
                logger.debug(f"Term '{term}': no hits found")

        except Exception as e:
            # Better error handling
            logger.warning(f"Error scoring term '{term}': {e}")
            term_scores[term] = 0.0

    return term_scores
    '''

    print("Replace your current implementation with this corrected version:")
    print(corrected_code)


# IMMEDIATE DEBUGGING SCRIPT
# ==========================

if __name__ == "__main__":
    print("🚨 DEBUGGING COLLECTION-LEVEL BM25")
    print("=" * 50)

    # Run step-by-step debugging
    debug_collection_bm25_step_by_step()

    # Check common issues
    check_common_issues()

    # Show corrected implementation
    get_corrected_implementation()

    print("\n🎯 NEXT STEPS:")
    print("1. Run this debugging script")
    print("2. Identify which step fails")
    print("3. Fix the implementation based on the findings")
    print("4. Test again with the corrected version")