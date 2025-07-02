#!/usr/bin/env python3
"""
Simple script to match document IDs from run file to ir_datasets collection
"""

import ir_datasets
from src.utils.file_utils import load_trec_run
from tqdm import tqdm


def match_doc_ids():
    """Check exact document ID matching between run file and ir_datasets"""

    print("🔍 DOCUMENT ID MATCHING ANALYSIS")
    print("=" * 40)

    # Load run file
    run_file = "/home/user/QDER/robust04/data/all/title.BM25_RM3_TUNED.run"
    print(f"Loading run file: {run_file}")

    first_stage_runs = load_trec_run(run_file)
    print(f"✅ Loaded runs for {len(first_stage_runs)} queries")

    # Get sample of document IDs from run file
    run_doc_ids = set()
    sample_query = list(first_stage_runs.keys())[0]

    # Get first 50 doc IDs from first query
    for doc_id, score in first_stage_runs[sample_query][:50]:
        run_doc_ids.add(doc_id)

    print(f"\n📋 RUN FILE DOCUMENT IDs (first 50 from query {sample_query}):")
    run_doc_list = list(run_doc_ids)
    for i, doc_id in enumerate(run_doc_list[:10]):
        print(f"  {i + 1:2d}. {doc_id}")
    print(f"  ... and {len(run_doc_list) - 10} more")

    # Load ir_datasets collection
    print(f"\n📚 LOADING IR_DATASETS COLLECTION...")
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")

    collection_doc_ids = set()
    matches_found = []

    print("Scanning collection for matches...")

    # Scan through collection looking for matches
    for doc in tqdm(dataset.docs_iter(), total=528155, desc="Checking docs"):
        collection_doc_ids.add(doc.doc_id)

        # Check if this doc ID is in our run file
        if doc.doc_id in run_doc_ids:
            matches_found.append(doc.doc_id)
            print(f"  ✅ MATCH FOUND: {doc.doc_id}")

        # Show first few collection doc IDs for comparison
        if len(collection_doc_ids) <= 10:
            print(f"  Collection doc {len(collection_doc_ids):2d}: {doc.doc_id}")

    # Final analysis
    print(f"\n📊 FINAL ANALYSIS:")
    print(f"Run file doc IDs checked: {len(run_doc_ids)}")
    print(f"Collection doc IDs scanned: {len(collection_doc_ids)}")
    print(f"Matches found: {len(matches_found)}")

    if len(matches_found) > 0:
        print(f"✅ SUCCESS: Found {len(matches_found)} matching documents!")
        print(f"Matching doc IDs: {matches_found}")

        # Calculate success rate
        success_rate = len(matches_found) / len(run_doc_ids) * 100
        print(f"Match rate: {success_rate:.1f}%")

        if success_rate > 80:
            print("✅ Excellent match rate - collections are compatible")
        elif success_rate > 20:
            print("⚠️ Partial match - some documents missing")
        else:
            print("❌ Poor match rate - might be wrong collection")

    else:
        print(f"❌ ZERO MATCHES FOUND!")
        print(f"This confirms run file and ir_datasets use different document collections")

        # Show format comparison
        print(f"\n🔍 FORMAT COMPARISON:")
        print(f"Run file format examples:")
        for doc_id in list(run_doc_ids)[:3]:
            print(f"  - {doc_id} (length: {len(doc_id)})")

        print(f"Collection format examples:")
        for doc_id in list(collection_doc_ids)[:3]:
            print(f"  - {doc_id} (length: {len(doc_id)})")

    return len(matches_found), len(run_doc_ids)


def check_all_trec_collections():
    """Check multiple TREC collections to find the right one"""

    print("\n🔍 CHECKING MULTIPLE TREC COLLECTIONS")
    print("=" * 42)

    # Collections to try
    collections = [
        "disks45/nocr/trec-robust-2004",
        # "disks45/nocr",
        # "aquaint/trec-robust-2005",
        # "trec-robust-2004"  # Alternative name
    ]

    # Get sample doc IDs from run file
    run_file = "/home/user/QDER/robust04/data/all/title.BM25_RM3_TUNED.run"
    first_stage_runs = load_trec_run(run_file)
    sample_query = list(first_stage_runs.keys())[0]

    run_doc_ids = set()
    for doc_id, score in first_stage_runs[sample_query][:20]:  # Check first 20
        run_doc_ids.add(doc_id)

    print(f"Looking for these doc IDs: {list(run_doc_ids)[:5]}...")

    for collection_name in collections:
        try:
            print(f"\n📚 Checking: {collection_name}")
            dataset = ir_datasets.load(collection_name)

            matches = 0
            docs_checked = 0

            # Check first 10K docs for speed
            for doc in dataset.docs_iter():
                if doc.doc_id in run_doc_ids:
                    matches += 1
                    print(f"  ✅ Found: {doc.doc_id}")

                docs_checked += 1
                if docs_checked >= 10000:  # Check first 10K for speed
                    break

            print(f"  Result: {matches}/{len(run_doc_ids)} matches in first {docs_checked} docs")

            if matches > 0:
                print(f"  🎯 POTENTIAL MATCH FOUND IN: {collection_name}")

                # If we found matches, scan the full collection
                if matches < len(run_doc_ids):
                    print(f"  Scanning full collection...")
                    total_matches = 0
                    for doc in tqdm(dataset.docs_iter(), desc=f"Scanning {collection_name}"):
                        if doc.doc_id in run_doc_ids:
                            total_matches += 1

                    print(f"  Final result: {total_matches}/{len(run_doc_ids)} total matches")

                return collection_name

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n❌ No matching collection found in any of: {collections}")
    return None


if __name__ == "__main__":
    # First check the default collection
    matches, total = match_doc_ids()

    # If no matches, try other collections
    if matches == 0:
        print("\n" + "=" * 50)
        print("No matches found - trying other TREC collections...")
        matching_collection = check_all_trec_collections()

        if matching_collection:
            print(f"\n🎉 SOLUTION FOUND!")
            print(f"Use this collection instead: '{matching_collection}'")
        else:
            print(f"\n❌ PROBLEM: Your run file doesn't match any standard TREC collection")
            print(f"   Possible issues:")
            print(f"   1. Run file is for a different TREC dataset")
            print(f"   2. Document IDs have been modified/transformed")
            print(f"   3. Custom document collection not in ir_datasets")
    else:
        print(f"\n🎉 SUCCESS!")
        print(f"Found {matches}/{total} matching documents")
        print(f"Your evaluation pipeline should work with full document loading")