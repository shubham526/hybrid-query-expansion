#!/usr/bin/env python3
"""
Debug why evaluation function returns all zeros
"""

# ISSUE 1: DOCUMENT TEXT FIELD
# ============================

# QUICK FIX 1: Check what field exists
def debug_document_fields():
    """Check what fields exist in TREC Robust documents"""
    import ir_datasets

    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")

    # Check first document
    for doc in dataset.docs_iter():
        print(f"Document ID: {doc.doc_id}")
        print(f"Available attributes: {dir(doc)}")

        if hasattr(doc, 'text'):
            print(f"doc.text: '{doc.text[:100]}...'")
        if hasattr(doc, 'body'):
            print(f"doc.body: '{doc.body[:100]}...'")

        # Try the correct field
        doc_content = doc.text if hasattr(doc, 'text') else doc.body
        print(f"Selected content: '{doc_content[:100]}...'")
        break


# ISSUE 2: NO CANDIDATE DOCUMENTS
# ===============================

# Your diagnostic shows "1 candidates" - this is way too few!
# A proper evaluation should have 50-1000 candidates per query

def debug_candidate_count():
    """Debug why so few candidates per query"""

    # Load your run file
    from src.utils.file_utils import load_trec_run
    run_file = "/home/user/QDER/robust04/data/all/title.BM25_RM3_TUNED.run"
    first_stage_runs = load_trec_run(run_file)

    # Check how many candidates per query
    print("🔍 CANDIDATE COUNT ANALYSIS")
    print("=" * 30)

    for qid, candidates in list(first_stage_runs.items())[:5]:
        print(f"Query {qid}: {len(candidates)} candidates")

        # Show first few
        for i, (doc_id, score) in enumerate(candidates[:3]):
            print(f"  {i + 1}. {doc_id} (score: {score})")

    # Check if document IDs match your document collection
    print("\n🔍 DOCUMENT ID MATCHING")
    print("=" * 25)

    import ir_datasets
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")

    # Load some documents
    doc_ids_in_collection = set()
    for i, doc in enumerate(dataset.docs_iter()):
        doc_ids_in_collection.add(doc.doc_id)
        if i > 1000:  # Check first 1000
            break

    # Check overlap with run file
    sample_query = list(first_stage_runs.keys())[0]
    run_doc_ids = {doc_id for doc_id, _ in first_stage_runs[sample_query][:10]}

    overlap = len(run_doc_ids & doc_ids_in_collection)
    print(f"Sample query {sample_query}:")
    print(f"  Run doc IDs: {list(run_doc_ids)[:5]}")
    print(f"  Collection has {len(doc_ids_in_collection)} docs")
    print(f"  Overlap: {overlap}/{len(run_doc_ids)} docs found")

    if overlap == 0:
        print("🚨 CRITICAL: No overlap between run file and document collection!")
        print("   → Document IDs don't match")


# ISSUE 3: QRELS MISMATCH
# =======================

def debug_qrels_matching():
    """Check if qrels match your queries and documents"""

    import ir_datasets
    from collections import defaultdict

    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")

    # Load qrels
    all_qrels = defaultdict(dict)
    qrel_count = 0
    for qrel in dataset.qrels_iter():
        all_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        qrel_count += 1
        if qrel_count > 100:  # Sample
            break

    print("🔍 QRELS ANALYSIS")
    print("=" * 17)

    sample_qid = list(all_qrels.keys())[0]
    sample_qrels = all_qrels[sample_qid]

    print(f"Sample query {sample_qid}:")
    print(f"  Has {len(sample_qrels)} relevance judgments")

    relevant_docs = [doc_id for doc_id, rel in sample_qrels.items() if rel > 0]
    print(f"  {len(relevant_docs)} relevant documents")

    if relevant_docs:
        print(f"  Sample relevant doc: {relevant_docs[0]}")
    else:
        print("  ❌ No relevant documents found!")


# QUICK FIX SCRIPT
# ================

def quick_fix_evaluation_test():
    """Quick test with all fixes applied"""

    print("🔧 TESTING EVALUATION WITH FIXES")
    print("=" * 35)

    # Your imports...
    from src.core.rm_expansion import RMExpansion
    from src.core.semantic_similarity import SemanticSimilarity
    from src.models.memory_efficient_reranker import create_memory_efficient_reranker
    from src.evaluation.evaluator import create_trec_dl_evaluator
    from src.utils.lucene_utils import initialize_lucene
    from src.utils.file_utils import load_json, load_trec_run
    from collections import defaultdict
    import ir_datasets

    # Initialize
    index_path = "/home/user/hybrid-query-expansion/robust/index_all-MiniLM-L6-v2/disks45_nocr_trec-robust-2004_sentence-transformers_all-MiniLM-L6-v2/"
    lucene_path = "/home/user/lucene-10.1.0/modules/"
    feature_file = "/home/user/hybrid-query-expansion/robust/experiments/robust2004_in_domain/features/fold4/disks45_nocr_trec-robust-2004_fold_4_train_qids_features.json.gz"
    run_file = "/home/user/QDER/robust04/data/all/title.BM25_RM3_TUNED.run"
    query_ids_file = "/home/user/hybrid-query-expansion/robust/experiments/robust2004_in_domain/temp_fold_files/fold_4_train_qids.txt"

    initialize_lucene(lucene_path)

    # Load data
    features = load_json(feature_file)
    with open(query_ids_file, 'r') as f:
        train_qids = {line.strip() for line in f if line.strip()}

    # Take 2 queries for quick test
    sample_qids = list(train_qids)[:2]

    # Load dataset with CORRECT document field
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
    all_queries = {q.query_id: q.title for q in dataset.queries_iter() if q.query_id in sample_qids}

    # FIX 1: Use correct document field
    all_documents = {}
    doc_count = 0
    for doc in dataset.docs_iter():
        # Use the correct field
        doc_content = doc.text if hasattr(doc, 'text') else doc.body
        all_documents[doc.doc_id] = doc_content
        doc_count += 1
        if doc_count > 5000:  # Limit for speed
            break

    print(f"✅ Loaded {len(all_documents)} documents using correct field")

    # FIX 2: Load qrels correctly
    all_qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        if qrel.query_id in sample_qids:
            all_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    print(f"✅ Loaded qrels for {len(all_qrels)} queries")

    # FIX 3: Load runs and check candidate count
    first_stage_runs = load_trec_run(run_file)
    first_stage_runs = {qid: run for qid, run in first_stage_runs.items() if qid in sample_qids}

    for qid, candidates in first_stage_runs.items():
        print(f"Query {qid}: {len(candidates)} candidates")

        # Check how many candidates have documents
        available_candidates = []
        for doc_id, score in candidates[:50]:  # Check top 50
            if doc_id in all_documents:
                available_candidates.append((doc_id, score))

        print(f"  {len(available_candidates)} candidates have document text")

        if len(available_candidates) == 0:
            print(f"  🚨 PROBLEM: No candidates have document text!")

        first_stage_runs[qid] = available_candidates

    # Create reranker
    reranker = create_memory_efficient_reranker("all-MiniLM-L6-v2", large_candidate_sets=False)

    # Test with RM-only (should work)
    print("\n🧪 TESTING RM-ONLY (α=1.0, β=0.0, γ=0.0)")

    reranked_runs = {}
    for qid, query_text in all_queries.items():
        if qid not in features or qid not in first_stage_runs:
            continue

        query_features = features[qid]

        # RM-only importance weights
        importance_weights = {
            term: term_data['rm_weight']  # Only RM component
            for term, term_data in query_features['term_features'].items()
        }

        # Expansion terms
        expansion_terms = [(term, term_data['rm_weight'])
                           for term, term_data in query_features['term_features'].items()]

        # Candidate results
        candidate_results = []
        for doc_id, first_stage_score in first_stage_runs[qid][:20]:  # Top 20
            doc_text = all_documents.get(doc_id, "")
            if doc_text:
                candidate_results.append((doc_id, doc_text, first_stage_score))

        print(f"  Query {qid}: {len(candidate_results)} candidates for reranking")

        if candidate_results:
            try:
                reranked_results = reranker.rerank_streaming(
                    query=query_text,
                    expansion_terms=expansion_terms,
                    importance_weights=importance_weights,
                    candidate_results=candidate_results,
                    top_k=20
                )
                reranked_runs[qid] = reranked_results
                print(f"    ✅ Reranked to {len(reranked_results)} results")
            except Exception as e:
                print(f"    ❌ Reranking error: {e}")
                reranked_runs[qid] = [(doc_id, score) for doc_id, _, score in candidate_results]
        else:
            print(f"    ❌ No candidates available")
            reranked_runs[qid] = []

    # Evaluate
    if reranked_runs:
        try:
            evaluator = create_trec_dl_evaluator()
            evaluation = evaluator.evaluate_run(reranked_runs, dict(all_qrels))
            score = evaluation.get("ndcg_cut_10", 0.0)
            print(f"\n🎯 RM-ONLY RESULT: nDCG@10 = {score:.4f}")

            if score > 0:
                print("✅ EVALUATION PIPELINE IS NOW WORKING!")
                print("   → The issue was document field and/or candidate matching")
            else:
                print("❌ Still getting zero score - need deeper debugging")

        except Exception as e:
            print(f"❌ Evaluation error: {e}")
    else:
        print("❌ No reranked runs to evaluate")


if __name__ == "__main__":
    # Run all debugging
    print("🔍 DEBUGGING ZERO EVALUATION SCORES")
    print("=" * 40)

    print("\n1. Checking document fields...")
    debug_document_fields()

    print("\n2. Checking candidate counts...")
    debug_candidate_count()

    print("\n3. Checking qrels matching...")
    debug_qrels_matching()

    print("\n4. Testing with fixes...")
    quick_fix_evaluation_test()