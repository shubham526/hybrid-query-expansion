#!/usr/bin/env python3
"""
Pre-computes and saves all features required for expansion weight learning.

This script iterates through a given dataset and for each query, computes and
stores the three core features for each potential expansion term:
1.  RM (Relevance Model) Weight
2.  BM25 Score
3.  Semantic Similarity Score

It uses the best available source for pseudo-relevance feedback (PRF) in the
following order of priority:
1. `scoreddocs` from the ir_datasets package.
2. A user-provided TREC-formatted run file.
3. The official `qrels` from the dataset (as a fallback).

Usage:
    # For a dataset with scoreddocs (e.g., TREC DL)
    python scripts/create_training_data.py \
        --dataset msmarco-passage/trec-dl-2019 \
        --output_dir ./training_data \
        --index_path ./indexes/msmarco-passage_bert-base-uncased \
        --lucene_path /path/to/your/lucene/jars/

    # For a dataset without scoreddocs, using a custom run file for PRF
    python scripts/create_training_data.py \
        --dataset disks45/nocr/trec-robust-2004 \
        --run_file_path ./my_bm25_runs/robust04.txt \
        --output_dir ./training_data_robust \
        --index_path ./indexes/disks45_nocr_trec-robust-2004_bert-base-uncased \
        --lucene_path /path/to/your/lucene/jars/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import traceback

# Add project root to path for local imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import ir_datasets
from tqdm import tqdm

# Import your project's modules
from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.utils.file_utils import save_json, load_trec_run, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation

# Conditionally import BM25 infrastructure
try:
    from src.core.bm25_scorer import TokenBM25Scorer
    from src.utils.initialize_lucene import initialize_lucene

    BM25_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BM25 scorer components not available, BM25 features will be zero. Error: {e}")
    TokenBM25Scorer = None
    initialize_lucene = None
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts and saves features for query expansion weight learning.
    """

    def __init__(self,
                 rm_expansion: RMExpansion,
                 semantic_similarity: SemanticSimilarity,
                 bm25_scorer: Optional[TokenBM25Scorer] = None,
                 max_expansion_terms: int = 20,
                 top_k_pseudo_docs: int = 10):
        self.rm_expansion = rm_expansion
        self.semantic_sim = semantic_similarity
        self.bm25_scorer = bm25_scorer
        self.max_expansion_terms = max_expansion_terms
        self.top_k_pseudo_docs = top_k_pseudo_docs
        logger.info("FeatureExtractor initialized.")

    def process_dataset(self,
                        full_dataset_name: str,
                        queries_to_process: Dict[str, str],
                        run_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a given subset of queries to extract features.
        """
        logger.info(f"Loading full dataset '{full_dataset_name}' to access docs and qrels...")
        dataset = ir_datasets.load(full_dataset_name)

        with TimedOperation(logger, "Loading all dataset components into memory"):
            qrels = {q.query_id: {d.doc_id: d.relevance for d in q.qrels_iter()} for q in dataset.queries_iter()}
            docs = {d.doc_id: d.text for d in dataset.docs_iter()}

        prf_source = defaultdict(list)

        # Determine the source for pseudo-relevance feedback
        if dataset.has_scoreddocs():
            logger.info("Found 'scoreddocs' in dataset. Using them for pseudo-relevance feedback.")
            for sdoc in dataset.scoreddocs_iter():
                prf_source[sdoc.query_id].append((sdoc.doc_id, sdoc.score))
        elif run_file_path and Path(run_file_path).exists():
            logger.info(f"Using user-provided run file at '{run_file_path}' for pseudo-relevance feedback.")
            loaded_run = load_trec_run(run_file_path)
            for qid, results in loaded_run.items():
                prf_source[qid] = results
        elif dataset.has_qrels():
            logger.warning(f"Dataset has no 'scoreddocs' and no run file was provided.")
            logger.warning("Falling back to using 'qrels' for PRF. This is an ORACLE experiment.")
            for qid, qrel_data in qrels.items():
                positive_docs = [(doc_id, rel) for doc_id, rel in qrel_data.items() if rel > 0]
                positive_docs.sort(key=lambda x: x[1], reverse=True)
                prf_source[qid] = positive_docs
        else:
            raise ValueError(
                f"Dataset '{full_dataset_name}' has no usable source for PRF (no scoreddocs, qrels, or provided run file).")

        all_query_features = {}
        failed_queries_count = 0

        logger.info(f"Starting feature extraction for {len(queries_to_process)} queries...")
        for query_id, query_text in tqdm(queries_to_process.items(), desc="Extracting Features"):
            try:
                if query_id not in prf_source or not prf_source[query_id]:
                    logger.warning(f"Skipping query {query_id}: No PRF documents found.")
                    failed_queries_count += 1
                    continue

                top_docs_for_prf = prf_source[query_id][:self.top_k_pseudo_docs]
                pseudo_docs_text = [docs.get(doc_id, "") for doc_id, _ in top_docs_for_prf]
                pseudo_scores = [score for _, score in top_docs_for_prf]

                if not any(pseudo_docs_text):
                    logger.warning(f"Skipping query {query_id}: Text for PRF documents not found.")
                    failed_queries_count += 1
                    continue

                features = self._extract_features_for_query(query_id, query_text, top_docs_for_prf, pseudo_docs_text,
                                                            pseudo_scores)
                if features:
                    all_query_features[query_id] = features

            except Exception as e:
                logger.error(f"Failed to process query {query_id}: {e}")
                logger.debug(traceback.format_exc())
                failed_queries_count += 1

        logger.info(f"Feature extraction complete. Successfully processed {len(all_query_features)} queries.")
        if failed_queries_count > 0:
            logger.warning(f"{failed_queries_count} queries failed or were skipped.")

        return all_query_features

    def _extract_features_for_query(self,
                                    query_id: str,
                                    query_text: str,
                                    top_docs_for_prf: List[Tuple[str, float]],
                                    pseudo_docs: List[str],
                                    pseudo_scores: List[float]) -> Optional[Dict[str, Any]]:
        """Extracts all features for a single query."""
        expansion_terms_with_weights = self.rm_expansion.expand_query(
            query=query_text, documents=pseudo_docs, scores=pseudo_scores,
            num_expansion_terms=self.max_expansion_terms, rm_type="rm3"
        )
        if not expansion_terms_with_weights:
            return None

        expansion_words = [term for term, _ in expansion_terms_with_weights]
        semantic_scores = self.semantic_sim.compute_query_expansion_similarities(query_text, expansion_words)

        reference_doc_id = top_docs_for_prf[0][0] if top_docs_for_prf else None
        bm25_scores = {}
        if self.bm25_scorer and reference_doc_id:
            try:
                bm25_scores = self.bm25_scorer.compute_bm25_term_weight(reference_doc_id, expansion_words)
            except Exception as e:
                logger.warning(f"BM25 scoring failed for query {query_id}, doc {reference_doc_id}: {e}")

        term_features = {
            term: {
                'rm_weight': float(rm_weight),
                'bm25_score': float(bm25_scores.get(term, 0.0)),
                'semantic_score': float(semantic_scores.get(term, 0.0))
            } for term, rm_weight in expansion_terms_with_weights
        }

        return {'query_text': query_text, 'term_features': term_features, 'reference_doc_id': reference_doc_id}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute and save features for query expansion weight learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Main Arguments ---
    parser.add_argument('--dataset', type=str, required=True, help='Name of the ir_datasets dataset to process.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the output feature file.')

    # --- PRF Source Arguments ---
    parser.add_argument('--run-file-path', type=str, default=None,
                        help='Optional path to a TREC run file to use for PRF if scoreddocs are unavailable.')

    # --- Filtering and Model Arguments ---
    parser.add_argument('--query-ids-file', type=str, default=None,
                        help='Optional path to a file with newline-separated query IDs to process (for k-fold CV).')
    parser.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model for semantic similarity.')
    parser.add_argument('--max-expansion-terms', type=int, default=20,
                        help='Maximum number of expansion terms to featurize.')

    # --- BM25 Arguments ---
    parser.add_argument('--index-path', type=str, default=None, help='Path to the pre-built BM25 index.')
    parser.add_argument('--lucene-path', type=str, default=None, help='Path to Lucene JAR files.')

    # --- Logging ---
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("create_training_data", args.log_level,
                                      str(output_dir / 'feature_extraction.log'))
    log_experiment_info(logger, **vars(args))

    try:
        # --- Initialize Components ---
        with TimedOperation(logger, "Initializing components"):
            rm_expansion = RMExpansion()
            semantic_sim = SemanticSimilarity(args.semantic_model)
            bm25_scorer = None
            if args.index_path:
                if not BM25_AVAILABLE: raise ImportError("BM25 components requested but not available.")
                if not args.lucene_path: raise ValueError("--lucene_path is required for BM25.")
                initialize_lucene(args.lucene_path)
                bm25_scorer = TokenBM25Scorer(args.index_path)

        # --- Determine which queries to process ---
        full_dataset = ir_datasets.load(args.dataset)
        queries_to_process = {q.query_id: q.text for q in full_dataset.queries_iter()}
        if args.query_ids_file:
            logger.info(f"Filtering queries using subset from: {args.query_ids_file}")
            with open(args.query_ids_file, 'r') as f:
                subset_qids = {line.strip() for line in f if line.strip()}
            queries_to_process = {qid: text for qid, text in queries_to_process.items() if qid in subset_qids}
        logger.info(f"Will process {len(queries_to_process)} queries.")

        # --- Run Feature Extraction ---
        extractor = FeatureExtractor(rm_expansion, semantic_sim, bm25_scorer, args.max_expansion_terms)
        all_features = extractor.process_dataset(args.dataset, queries_to_process, args.run_file_path)

        # --- Save Results ---
        if all_features:
            dataset_name_safe = args.dataset.replace('/', '_')
            subset_name = Path(args.query_ids_file).stem if args.query_ids_file else "full"
            output_filename = f"{dataset_name_safe}_{subset_name}_features.json.gz"
            output_file_path = output_dir / output_filename

            logger.info(f"Saving {len(all_features)} queries' features to {output_file_path}...")
            save_json(all_features, output_file_path, compress=True)
            logger.info("Features saved successfully.")
        else:
            logger.warning("No features were extracted. No output file was created.")

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 60 + "\nFEATURE EXTRACTION SCRIPT COMPLETED SUCCESSFULLY\n" + "=" * 60)


if __name__ == "__main__":
    main()