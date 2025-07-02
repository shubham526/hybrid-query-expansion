#!/usr/bin/env python3
"""
Pre-computes and saves all features required for expansion weight learning.

This script iterates through a given dataset and for each query, computes and
stores the three core features for each potential expansion term:
1.  RM (Relevance Model) Weight
2.  BM25 Score
3.  Semantic Similarity Score

It is designed to be flexible and robust to variations in ir_datasets formats.
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterator
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
    logging.error(f"Failed to import from 'src'. Please run 'pip install -e .' from the project root. Error: {e}")
    INDEXER_AVAILABLE = False
    TokenBM25Scorer = None
    initialize_lucene = None
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_query_text(query_obj: Any) -> str:
    """
    Flexibly extracts query text from different ir_datasets query types.
    Handles MS MARCO (text) and TREC (title, description) formats.
    """
    if hasattr(query_obj, 'text'):
        return query_obj.text
    elif hasattr(query_obj, 'title'):
        if hasattr(query_obj, 'description') and query_obj.description:
            return f"{query_obj.title} {query_obj.description}"
        return query_obj.title
    else:
        logger.warning(f"Could not determine query text for query_id {query_obj.query_id}. Defaulting to empty string.")
        return ""


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
        """Processes a given subset of queries to extract features."""
        logger.info(f"Loading full dataset '{full_dataset_name}' to access docs and qrels...")
        dataset = ir_datasets.load(full_dataset_name)

        with TimedOperation(logger, "Loading all dataset components into memory"):
            # --- FIX: Load qrels from the top-level dataset iterator ---
            qrels = defaultdict(dict)
            if dataset.has_qrels():
                for qrel in dataset.qrels_iter():
                    qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

            docs = {doc.doc_id: (doc.text if hasattr(doc, 'text') else doc.body) for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count(), desc="Loading docs" )}

        prf_source = defaultdict(list)
        if dataset.has_scoreddocs():
            logger.info("Found 'scoreddocs' in dataset. Using them for pseudo-relevance feedback.")
            for sdoc in dataset.scoreddocs_iter():
                prf_source[sdoc.query_id].append((sdoc.doc_id, sdoc.score))
        elif run_file_path and Path(run_file_path).exists():
            logger.info(f"Using user-provided run file at '{run_file_path}' for PRF.")
            prf_source.update(load_trec_run(run_file_path))
        elif dataset.has_qrels():
            logger.warning(
                f"No 'scoreddocs' or run file provided. Falling back to 'qrels' for PRF (ORACLE experiment).")
            for qid, qrel_data in qrels.items():
                prf_source[qid] = sorted([(doc_id, rel) for doc_id, rel in qrel_data.items() if rel > 0],
                                         key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Dataset '{full_dataset_name}' has no usable source for PRF.")

        all_query_features = {}
        for query_id, query_text in tqdm(queries_to_process.items(), desc="Extracting Features"):
            try:
                if query_id not in prf_source or not prf_source[query_id]: continue

                top_docs_for_prf = prf_source[query_id][:self.top_k_pseudo_docs]
                pseudo_docs_text = [docs.get(doc_id, "") for doc_id, _ in top_docs_for_prf]
                if not any(pseudo_docs_text): continue

                features = self._extract_features_for_query(query_id, query_text, top_docs_for_prf, pseudo_docs_text)
                if features: all_query_features[query_id] = features
            except Exception as e:
                logger.error(f"Failed to process query {query_id}: {e}", exc_info=True)

        return all_query_features

    def _extract_features_for_query(self, query_id, query_text, top_docs_for_prf, pseudo_docs) -> Optional[
        Dict[str, Any]]:
        """Extracts all features for a single query."""
        pseudo_scores = [score for _, score in top_docs_for_prf]
        expansion_terms = self.rm_expansion.expand_query(query_text, pseudo_docs, pseudo_scores,
                                                         self.max_expansion_terms)
        if not expansion_terms: return None

        expansion_words = [term for term, _ in expansion_terms]
        semantic_scores = self.semantic_sim.compute_query_expansion_similarities(query_text, expansion_words)

        reference_doc_id = top_docs_for_prf[0][0] if top_docs_for_prf else None
        bm25_scores = {}
        if self.bm25_scorer and reference_doc_id:
            try:
                bm25_scores = self.bm25_scorer.compute_bm25_term_weight(reference_doc_id, expansion_words)
            except Exception as e:
                logger.warning(f"BM25 scoring failed for query {query_id}, doc {reference_doc_id}: {e}")

        term_features = {
            term: {'rm_weight': float(rm_w), 'bm25_score': float(bm25_scores.get(term, 0.0)),
                   'semantic_score': float(semantic_scores.get(term, 0.0))}
            for term, rm_w in expansion_terms
        }
        return {'query_text': query_text, 'term_features': term_features, 'reference_doc_id': reference_doc_id}


def main():
    parser = argparse.ArgumentParser(description="Pre-compute features for query expansion weight learning.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, help='Name of the ir_datasets dataset.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the feature file.')
    parser.add_argument('--run-file-path', type=str, help='Optional path to a TREC run file for PRF.')
    parser.add_argument('--query-ids-file', type=str, help='Optional path to a file with query IDs to process.')
    parser.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2', help='Sentence-transformer model.')
    parser.add_argument('--max-expansion-terms', type=int, default=20, help='Max expansion terms.')
    parser.add_argument('--index-path', type=str, help='Path to BM25 index.')
    parser.add_argument('--lucene-path', type=str, help='Path to Lucene JARs.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("create_training_data", args.log_level,
                                      str(output_dir / 'feature_extraction.log'))
    log_experiment_info(logger, **vars(args))

    try:
        with TimedOperation(logger, "Initializing components"):
            rm_expansion = RMExpansion()
            semantic_sim = SemanticSimilarity(args.semantic_model)
            bm25_scorer = None
            if args.index_path:
                if not BM25_AVAILABLE: raise ImportError("BM25 components not available.")
                if not args.lucene_path: raise ValueError("--lucene-path is required for BM25.")
                initialize_lucene(args.lucene_path)
                bm25_scorer = TokenBM25Scorer(args.index_path)

        dataset = ir_datasets.load(args.dataset)
        all_queries = {q.query_id: get_query_text(q) for q in dataset.queries_iter()}

        queries_to_process = all_queries
        if args.query_ids_file:
            logger.info(f"Filtering queries using subset from: {args.query_ids_file}")
            with open(args.query_ids_file, 'r') as f:
                subset_qids = {line.strip() for line in f if line.strip()}
            queries_to_process = {qid: text for qid, text in all_queries.items() if qid in subset_qids}

        logger.info(f"Will process {len(queries_to_process)} queries.")

        extractor = FeatureExtractor(rm_expansion, semantic_sim, bm25_scorer, args.max_expansion_terms)
        all_features = extractor.process_dataset(args.dataset, queries_to_process, args.run_file_path)

        if all_features:
            safe_name = args.dataset.replace('/', '_')
            subset_name = Path(args.query_ids_file).stem if args.query_ids_file else "full"
            output_filename = f"{safe_name}_{subset_name}_features.json"
            output_path = output_dir / output_filename
            logger.info(f"Saving features to {output_path}...")
            save_json(all_features, output_path, compress=True)
            logger.info("Features saved successfully.")
        else:
            logger.warning("No features were extracted.")

    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 60 + "\nFEATURE EXTRACTION SCRIPT COMPLETED\n" + "=" * 60)


if __name__ == "__main__":
    main()