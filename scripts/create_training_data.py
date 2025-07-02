#!/usr/bin/env python3
"""
Train importance weights for query expansion.

This script learns optimal weights (alpha, beta, gamma) for combining:
- RM weights
- BM25 scores
- Semantic similarity scores

The weights are optimized to maximize retrieval performance on a validation set.

Usage:
    python train_weights.py --training_data ./training_data --validation_runs ./trec_dl_runs --output_dir ./models
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import ir_datasets
from tqdm import tqdm

# Import project modules
from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.models.weight_optimizer import LBFGSOptimizer, GridSearchOptimizer, create_optimizer
from src.models.multivector_reranking import TRECDLReranker
from src.evaluation.evaluator import create_trec_dl_evaluator
from src.utils.file_utils import (load_training_data, save_learned_weights,
                              save_experiment_results, ensure_dir)
from src.utils.logging_utils import (setup_experiment_logging, log_experiment_info,
                                 log_weight_optimization, TimedOperation)

# Import existing BM25 infrastructure
try:
    from bert_bm25_scorer import TokenBM25Scorer, setup_lucene

    BM25_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BM25 scorer not available: {e}")
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


class WeightTrainer:
    """
    Trains importance weights for query expansion.
    """

    def __init__(self,
                 bm25_scorer: Optional[Any] = None,
                 semantic_similarity: Optional[SemanticSimilarity] = None,
                 rm_expansion: Optional[RMExpansion] = None,
                 reranker: Optional[TRECDLReranker] = None):
        """
        Initialize weight trainer.

        Args:
            bm25_scorer: BM25 scorer instance
            semantic_similarity: Semantic similarity computer
            rm_expansion: RM expansion instance
            reranker: Multi-vector reranker
        """
        self.bm25_scorer = bm25_scorer
        self.semantic_sim = semantic_similarity or SemanticSimilarity()
        self.rm_expansion = rm_expansion or RMExpansion()
        self.reranker = reranker

        logger.info("WeightTrainer initialized")
        logger.info(f"  BM25 available: {self.bm25_scorer is not None}")
        logger.info(f"  Reranker available: {self.reranker is not None}")

    def load_validation_data(self, dataset_name: str = "msmarco-passage/trec-dl-2019") -> Dict[str, Any]:
        """
        Load validation data from TREC DL.

        Args:
            dataset_name: TREC DL dataset identifier

        Returns:
            Dictionary with queries, qrels, and first-stage runs
        """
        logger.info(f"Loading validation data: {dataset_name}")

        dataset = ir_datasets.load(dataset_name)

        # Load queries
        queries = {q.query_id: q.text for q in dataset.queries_iter()}
        logger.info(f"Loaded {len(queries)} validation queries")

        # Load qrels
        qrels = defaultdict(dict)
        for qrel in dataset.qrels_iter():
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        logger.info(f"Loaded qrels for {len(qrels)} queries")

        # Load first-stage runs
        first_stage_runs = defaultdict(list)
        for scoreddoc in dataset.scoreddocs_iter():
            first_stage_runs[scoreddoc.query_id].append((scoreddoc.doc_id, scoreddoc.score))
        logger.info(f"Loaded first-stage runs for {len(first_stage_runs)} queries")

        # Load document collection for reranking
        logger.info("Loading document collection...")
        documents = {}
        with TimedOperation(logger, "Document loading"):
            for doc in tqdm(dataset.docs_iter(), desc="Loading docs"):
                documents[doc.doc_id] = doc.text
        logger.info(f"Loaded {len(documents)} documents")

        return {
            'queries': dict(queries),
            'qrels': dict(qrels),
            'first_stage_runs': dict(first_stage_runs),
            'documents': documents
        }

    def compute_expansion_terms_dict(self, queries: Dict[str, str],
                                     first_stage_runs: Dict[str, List[Tuple[str, float]]],
                                     documents: Dict[str, str],
                                     top_k_pseudo_docs: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute RM expansion terms for all validation queries.

        Args:
            queries: Query dictionary
            first_stage_runs: First-stage retrieval results
            documents: Document collection
            top_k_pseudo_docs: Number of top docs to use for pseudo-relevance

        Returns:
            Dictionary mapping query_id to expansion terms
        """
        logger.info("Computing RM expansion terms for validation queries...")

        expansion_terms_dict = {}
        failed_queries = []

        for query_id, query_text in tqdm(queries.items(), desc="Computing RM expansion"):
            try:
                if query_id not in first_stage_runs:
                    logger.warning(f"No first-stage run for query {query_id}")
                    continue

                # Get top pseudo-relevant documents
                top_docs = first_stage_runs[query_id][:top_k_pseudo_docs]
                pseudo_docs = []
                pseudo_scores = []

                for doc_id, score in top_docs:
                    if doc_id in documents:
                        pseudo_docs.append(documents[doc_id])
                        pseudo_scores.append(score)

                if not pseudo_docs:
                    logger.warning(f"No pseudo-relevant docs found for query {query_id}")
                    continue

                # RM expansion
                expansion_terms = self.rm_expansion.expand_query(
                    query=query_text,
                    documents=pseudo_docs,
                    scores=pseudo_scores,
                    num_expansion_terms=15,
                    rm_type="rm3"
                )

                if expansion_terms:
                    expansion_terms_dict[query_id] = expansion_terms
                else:
                    failed_queries.append(query_id)

            except Exception as e:
                logger.warning(f"RM expansion failed for query {query_id}: {e}")
                failed_queries.append(query_id)

        logger.info(f"Computed expansion terms for {len(expansion_terms_dict)} queries")
        if failed_queries:
            logger.warning(f"Failed to compute expansion for {len(failed_queries)} queries")

        return expansion_terms_dict

    def create_evaluation_function(self, validation_data: Dict[str, Any],
                                   expansion_terms_dict: Dict[str, List[Tuple[str, float]]],
                                   metric: str = "ndcg_cut_10") -> Callable:
        """
        Create evaluation function for weight optimization.

        Args:
            validation_data: Validation dataset
            expansion_terms_dict: Pre-computed expansion terms
            metric: Evaluation metric to optimize

        Returns:
            Evaluation function that takes weights and returns performance
        """
        queries = validation_data['queries']
        qrels = validation_data['qrels']
        first_stage_runs = validation_data['first_stage_runs']
        documents = validation_data['documents']

        def evaluate_weights(weights: Tuple[float, float, float]) -> float:
            """
            Evaluate retrieval performance with given weights.

            Args:
                weights: (alpha, beta, gamma) tuple

            Returns:
                Average performance score
            """
            alpha, beta, gamma = weights

            try:
                # Compute importance weights for all queries
                importance_weights_dict = {}

                for query_id, query_text in queries.items():
                    if query_id not in expansion_terms_dict:
                        continue

                    expansion_terms = expansion_terms_dict[query_id]
                    importance_weights = {}

                    # Get reference document for BM25 scoring
                    reference_doc_id = None
                    if query_id in first_stage_runs and first_stage_runs[query_id]:
                        reference_doc_id = first_stage_runs[query_id][0][0]  # Top document

                    # Compute importance for each expansion term
                    expansion_words = [term for term, weight in expansion_terms]

                    # Batch compute semantic similarities
                    semantic_scores = self.semantic_sim.compute_query_expansion_similarities(
                        query_text, expansion_words
                    )

                    for term, rm_weight in expansion_terms:
                        # BM25 score
                        bm25_score = 0.0
                        if self.bm25_scorer and reference_doc_id:
                            try:
                                bm25_scores = self.bm25_scorer.compute_bm25_term_weight(reference_doc_id, [term])
                                bm25_score = float(bm25_scores.get(term, 0.0))
                            except Exception:
                                bm25_score = 0.0

                        # Semantic score
                        semantic_score = semantic_scores.get(term, 0.0)

                        # Compute importance using current weights
                        importance = alpha * rm_weight + beta * bm25_score + gamma * semantic_score
                        importance_weights[term] = importance

                    importance_weights_dict[query_id] = importance_weights

                # Rerank using importance weights
                if self.reranker:
                    reranked_results = self.reranker.rerank_trec_dl_run(
                        queries=queries,
                        first_stage_runs=first_stage_runs,
                        expansion_terms_dict=expansion_terms_dict,
                        importance_weights_dict=importance_weights_dict,
                        top_k=100
                    )
                else:
                    # Fallback: use first-stage runs (no improvement expected)
                    logger.warning("No reranker available, using first-stage runs")
                    reranked_results = first_stage_runs

                # Evaluate performance
                evaluator = create_trec_dl_evaluator()
                evaluation = evaluator.evaluate_run(reranked_results, qrels)

                return evaluation.get(metric, 0.0)

            except Exception as e:
                logger.warning(f"Evaluation failed with weights {weights}: {e}")
                return 0.0

        return evaluate_weights

    def train_weights(self, training_data: Dict[str, Any], validation_data: Dict[str, Any],
                      optimizer_type: str = "lbfgs", metric: str = "ndcg_cut_10") -> Tuple[float, float, float]:
        """
        Train optimal importance weights.

        Args:
            training_data: Training dataset (for statistics/debugging)
            validation_data: Validation dataset for optimization
            optimizer_type: Type of optimizer ('lbfgs', 'grid', 'random')
            metric: Metric to optimize

        Returns:
            Optimal weights (alpha, beta, gamma)
        """
        logger.info(f"Training weights using {optimizer_type} optimizer")
        logger.info(f"Optimizing metric: {metric}")

        # Compute expansion terms for validation queries
        expansion_terms_dict = self.compute_expansion_terms_dict(
            validation_data['queries'],
            validation_data['first_stage_runs'],
            validation_data['documents']
        )

        # Create evaluation function
        evaluation_function = self.create_evaluation_function(
            validation_data, expansion_terms_dict, metric
        )

        # Test baseline performance (equal weights)
        logger.info("Evaluating baseline performance...")
        baseline_weights = (1.0, 1.0, 1.0)
        baseline_score = evaluation_function(baseline_weights)
        logger.info(f"Baseline performance ({metric}): {baseline_score:.4f}")

        # Initialize optimizer
        optimizer = create_optimizer(optimizer_type)

        # Optimize weights
        with TimedOperation(logger, "Weight optimization"):
            optimal_weights = optimizer.optimize_weights(
                training_data=training_data,  # Not directly used in L-BFGS, but kept for interface
                validation_queries=validation_data['queries'],
                validation_qrels=validation_data['qrels'],
                evaluation_function=evaluation_function
            )

        # Evaluate final performance
        final_score = evaluation_function(optimal_weights)

        # Log optimization results
        log_weight_optimization(
            logger,
            initial_weights=baseline_weights,
            final_weights=optimal_weights,
            initial_score=baseline_score,
            final_score=final_score,
            iterations=getattr(optimizer, 'iterations', 0)
        )

        return optimal_weights


def main():
    parser = argparse.ArgumentParser(description="Train importance weights for query expansion")

    # Data parameters
    parser.add_argument('--training_data', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--validation_dataset', type=str, default='msmarco-passage/trec-dl-2019',
                        help='Validation dataset name (default: msmarco-passage/trec-dl-2019)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for trained weights')

    # BM25 parameters
    parser.add_argument('--index_path', type=str, default=None,
                        help='Path to BM25 index (optional)')
    parser.add_argument('--lucene_path', type=str, default=None,
                        help='Path to Lucene JAR files (optional)')

    # Model parameters
    parser.add_argument('--semantic_model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence transformer model name')
    parser.add_argument('--optimizer', type=str, default='lbfgs',
                        choices=['lbfgs', 'grid', 'random'],
                        help='Optimization algorithm')
    parser.add_argument('--metric', type=str, default='ndcg_cut_10',
                        help='Metric to optimize (default: ndcg_cut_10)')

    # Logging parameters
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Setup logging
    logger = setup_experiment_logging("train_weights", args.log_level)

    # Log experiment configuration
    log_experiment_info(
        logger,
        training_data=args.training_data,
        validation_dataset=args.validation_dataset,
        output_dir=args.output_dir,
        semantic_model=args.semantic_model,
        optimizer=args.optimizer,
        metric=args.metric,
        index_path=args.index_path,
        bm25_available=BM25_AVAILABLE and args.index_path is not None
    )

    try:
        # Load training data
        logger.info("Loading training data...")
        training_data = load_training_data(args.training_data)
        logger.info(f"Loaded training data with {len(training_data.get('features', {}))} queries")

        # Initialize BM25 scorer if available
        bm25_scorer = None
        if args.index_path:
            logger.info(f'Loading BM25-BERT index from {args.index_path}')
            if args.lucene_path:
                try:
                    # Initialize JVM before importing Lucene-dependent modules
                    from src.utils.initialize_lucene import initialize_lucene
                    if not initialize_lucene(args.lucene_path):
                        logger.error("Failed to initialize Lucene")
                        sys.exit(1)

                    # Now initialize BM25 scorer
                    if BM25_AVAILABLE:
                        bm25_scorer = TokenBM25Scorer(args.index_path)
                        logger.info("BM25 scorer initialized successfully")
                    else:
                        logger.error("BM25 scorer not available despite successful Lucene initialization")
                        sys.exit(1)

                except Exception as e:
                    logger.error(f"Failed to initialize JVM: {str(e)}")
                    sys.exit(1)
            else:
                logger.error('ERROR: Must provide --lucene_path when using --index_path')
                sys.exit(1)
        else:
            logger.info('No BM25-BERT index provided. BM25 scores will default to 0.0')

        # Initialize semantic similarity
        logger.info(f"Initializing semantic similarity with model: {args.semantic_model}")
        semantic_sim = SemanticSimilarity(args.semantic_model)

        # Initialize RM expansion
        rm_expansion = RMExpansion()

        # Initialize reranker
        logger.info("Initializing multi-vector reranker...")
        reranker = TRECDLReranker(args.semantic_model)

        # Initialize trainer
        trainer = WeightTrainer(
            bm25_scorer=bm25_scorer,
            semantic_similarity=semantic_sim,
            rm_expansion=rm_expansion,
            reranker=reranker
        )

        # Load validation data
        validation_data = trainer.load_validation_data(args.validation_dataset)

        # Train weights
        optimal_weights = trainer.train_weights(
            training_data=training_data,
            validation_data=validation_data,
            optimizer_type=args.optimizer,
            metric=args.metric
        )

        # Save results
        output_dir = ensure_dir(args.output_dir)

        # Save learned weights
        save_learned_weights(
            weights=optimal_weights,
            filepath=output_dir / 'learned_weights.json',
            experiment_info={
                'training_data': args.training_data,
                'validation_dataset': args.validation_dataset,
                'semantic_model': args.semantic_model,
                'optimizer': args.optimizer,
                'metric': args.metric,
                'bm25_available': bm25_scorer is not None
            }
        )

        # Save complete experiment results
        results = {
            'optimal_weights': {
                'alpha': optimal_weights[0],
                'beta': optimal_weights[1],
                'gamma': optimal_weights[2]
            },
            'experiment_config': {
                'training_data': args.training_data,
                'validation_dataset': args.validation_dataset,
                'semantic_model': args.semantic_model,
                'optimizer': args.optimizer,
                'metric': args.metric
            }
        }

        save_experiment_results(results, output_dir, 'weight_training')

        logger.info("Weight training completed successfully!")
        logger.info(
            f"Optimal weights: alpha={optimal_weights[0]:.3f}, beta={optimal_weights[1]:.3f}, gamma={optimal_weights[2]:.3f}")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Weight training failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()