#!/usr/bin/env python3
"""
Pre-computes and saves all features required for expansion weight learning.

This script iterates through a given dataset (e.g., TREC DL validation set),
and for each query, it computes and stores the three core features for each
potential expansion term:
1.  RM (Relevance Model) Weight
2.  BM25 Score
3.  Semantic Similarity Score

The output is a single, structured JSON file containing all pre-computed
features, which can then be used by the `train_weights.py` script for fast
and efficient weight optimization.

Usage:
    python scripts/create_training_data.py \
        --dataset msmarco-passage/trec-dl-2019 \
        --output_dir ./training_data \
        --index_path ./indexes/msmarco-passage_bert-base-uncased \
        --lucene_path /path/to/lucene/jars/*
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
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
from src.utils.file_utils import save_json, ensure_dir
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
    This class handles the entire data pre-computation pipeline.
    """

    def __init__(self,
                 rm_expansion: RMExpansion,
                 semantic_similarity: SemanticSimilarity,
                 bm25_scorer: Optional[TokenBM25Scorer] = None,
                 max_expansion_terms: int = 20,
                 top_k_pseudo_docs: int = 10):
        """
        Initializes the FeatureExtractor.

        Args:
            rm_expansion: An instance of the RMExpansion class.
            semantic_similarity: An instance of the SemanticSimilarity class.
            bm25_scorer: An optional instance of the TokenBM25Scorer class.
            max_expansion_terms: The maximum number of expansion terms to consider.
            top_k_pseudo_docs: The number of top documents to use for pseudo-relevance feedback.
        """
        self.rm_expansion = rm_expansion
        self.semantic_sim = semantic_similarity
        self.bm25_scorer = bm25_scorer
        self.max_expansion_terms = max_expansion_terms
        self.top_k_pseudo_docs = top_k_pseudo_docs

        logger.info("FeatureExtractor initialized.")
        logger.info(f"  Max expansion terms: {self.max_expansion_terms}")
        logger.info(f"  Top pseudo-relevant docs: {self.top_k_pseudo_docs}")
        logger.info(f"  BM25 scorer available: {self.bm25_scorer is not None}")

    def process_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Processes an entire dataset to extract features for all queries.

        Args:
            dataset_name: The name of the dataset from ir_datasets.

        Returns:
            A dictionary containing the extracted features and metadata.
        """
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = ir_datasets.load(dataset_name)

        # Load all necessary data components into memory first
        with TimedOperation(logger, "Loading all dataset components"):
            queries = {q.query_id: q.text for q in dataset.queries_iter()}
            qrels = {q.query_id: {d.doc_id: d.relevance for d in q.qrels_iter()} for q in dataset.queries_iter()}
            docs = {d.doc_id: d.text for d in dataset.docs_iter()}
            first_stage_runs = defaultdict(list)
            if dataset.has_scoreddocs():
                for sdoc in dataset.scoreddocs_iter():
                    first_stage_runs[sdoc.query_id].append((sdoc.doc_id, sdoc.score))
            else:
                logger.warning(f"Dataset '{dataset_name}' does not have pre-computed scoreddocs (first-stage runs).")

        all_query_features = {}
        failed_queries = 0

        logger.info(f"Starting feature extraction for {len(queries)} queries...")
        for query_id, query_text in tqdm(queries.items(), desc="Extracting Features"):
            try:
                # Use the first-stage run for pseudo-relevance feedback
                if query_id not in first_stage_runs:
                    logger.warning(f"Skipping query {query_id}: No first-stage run found.")
                    failed_queries += 1
                    continue

                top_docs = first_stage_runs[query_id][:self.top_k_pseudo_docs]
                pseudo_docs = [docs.get(doc_id, "") for doc_id, _ in top_docs]
                pseudo_scores = [score for _, score in top_docs]

                if not any(pseudo_docs):
                    logger.warning(f"Skipping query {query_id}: No pseudo-relevant documents found in the collection.")
                    failed_queries += 1
                    continue

                features = self._extract_features_for_query(query_id, query_text, top_docs, pseudo_docs, pseudo_scores)
                if features:
                    all_query_features[query_id] = features

            except Exception as e:
                logger.error(f"Failed to process query {query_id}: {e}")
                logger.debug(traceback.format_exc())
                failed_queries += 1

        logger.info(f"Feature extraction complete. Successfully processed {len(all_query_features)} queries.")
        if failed_queries > 0:
            logger.warning(f"{failed_queries} queries failed during processing.")

        return all_query_features

    def _extract_features_for_query(self,
                                    query_id: str,
                                    query_text: str,
                                    top_docs: List[Tuple[str, float]],
                                    pseudo_docs: List[str],
                                    pseudo_scores: List[float]) -> Optional[Dict[str, Any]]:
        """
        Extracts all features for a single query.
        """
        # --- Step 1: RM Expansion ---
        expansion_terms_with_weights = self.rm_expansion.expand_query(
            query=query_text,
            documents=pseudo_docs,
            scores=pseudo_scores,
            num_expansion_terms=self.max_expansion_terms,
            rm_type="rm3"
        )
        if not expansion_terms_with_weights:
            return None

        expansion_words = [term for term, _ in expansion_terms_with_weights]

        # --- Step 2: Semantic Similarity (in batch) ---
        semantic_scores = self.semantic_sim.compute_query_expansion_similarities(query_text, expansion_words)

        # --- Step 3: BM25 Scores (in batch, if available) ---
        reference_doc_id = top_docs[0][0] if top_docs else None
        bm25_scores = {}
        if self.bm25_scorer and reference_doc_id:
            try:
                bm25_scores = self.bm25_scorer.compute_bm25_term_weight(reference_doc_id, expansion_words)
            except Exception as e:
                logger.warning(f"BM25 scoring failed for query {query_id}, doc {reference_doc_id}: {e}")

        # --- Step 4: Assemble Features ---
        term_features = {}
        for term, rm_weight in expansion_terms_with_weights:
            term_features[term] = {
                'rm_weight': float(rm_weight),
                'bm25_score': float(bm25_scores.get(term, 0.0)),
                'semantic_score': float(semantic_scores.get(term, 0.0))
            }

        return {
            'query_text': query_text,
            'term_features': term_features,
            'reference_doc_id': reference_doc_id
        }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute and save features for query expansion weight learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Required Arguments ---
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the ir_datasets dataset to process (e.g., "msmarco-passage/trec-dl-2019").')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output feature file.')

    # --- Model & Component Arguments ---
    parser.add_argument('--semantic_model', type=str, default='all-MiniLM-L6-v2',
                        help='Name of the sentence-transformer model for semantic similarity.')
    parser.add_argument('--max_expansion_terms', type=int, default=20,
                        help='Maximum number of expansion terms to generate and featurize.')

    # --- BM25 Arguments (Optional) ---
    parser.add_argument('--index_path', type=str, default=None,
                        help='Path to the pre-built BM25 index. If not provided, BM25 scores will be 0.')
    parser.add_argument('--lucene_path', type=str, default=None,
                        help='Path to Lucene JAR files. Required if --index_path is used.')

    # --- Logging Arguments ---
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # --- Setup ---
    output_dir = ensure_dir(args.output_dir)
    logger = setup_experiment_logging("create_training_data", args.log_level,
                                      str(output_dir / 'feature_extraction.log'))

    log_experiment_info(logger, **vars(args))

    # --- Initialize Components ---
    try:
        with TimedOperation(logger, "Initializing components"):
            rm_expansion = RMExpansion()
            semantic_sim = SemanticSimilarity(args.semantic_model)

            bm25_scorer = None
            if args.index_path:
                if not BM25_AVAILABLE:
                    raise ImportError("BM25 components could not be imported. Please check dependencies.")
                if not args.lucene_path:
                    raise ValueError("--lucene_path is required when using --index_path.")

                logger.info("Initializing Lucene and BM25 scorer...")
                initialize_lucene(args.lucene_path)
                bm25_scorer = TokenBM25Scorer(args.index_path)

        # --- Run Feature Extraction ---
        extractor = FeatureExtractor(
            rm_expansion=rm_expansion,
            semantic_similarity=semantic_sim,
            bm25_scorer=bm25_scorer,
            max_expansion_terms=args.max_expansion_terms
        )

        all_features = extractor.process_dataset(args.dataset)

        # --- Save Results ---
        if all_features:
            output_file = output_dir / f"{args.dataset.replace('/', '_')}_features.json.gz"
            logger.info(f"Saving all computed features to {output_file}...")
            save_json(all_features, output_file, compress=True)
            logger.info("Features saved successfully.")
        else:
            logger.warning("No features were extracted. No output file was created.")

    except Exception as e:
        logger.critical(f"A critical error occurred during the feature extraction process: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("FEATURE EXTRACTION SCRIPT COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()