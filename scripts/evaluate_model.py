#!/usr/bin/env python3
"""
Evaluates a trained importance-weighted query expansion model on a test set.

This script performs the final evaluation by:
1.  Loading learned weights (alpha, beta, gamma) from a file.
2.  Running the full expansion and multi-vector reranking pipeline on a specified
    test dataset (e.g., TREC DL 2019/2020).
3.  Conducting a comprehensive ablation study to evaluate the contribution of each
    component (RM, BM25, semantic similarity).
4.  Generating TREC-formatted run files for each model configuration.
5.  Computing and saving final evaluation metrics (e.g., nDCG@10, MAP).

Usage:
    python scripts/evaluate_model.py \
        --weights_file ./models/learned_weights.json \
        --dataset msmarco-passage/trec-dl-2019 \
        --output_dir ./evaluation_results \
        --index_path ./indexes/msmarco-passage_bert-base-uncased \
        --lucene_path /path/to/lucene/jars/*
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
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
from src.models.expansion_models import create_baseline_comparison_models, ExpansionModel
from src.models.multivector_reranking import MultiVectorReranker
from src.evaluation.evaluator import TRECEvaluator
from src.utils.file_utils import load_learned_weights, save_json, save_trec_run, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation, log_results

# Conditionally import BM25 infrastructure
try:
    from src.core.bm25_scorer import TokenBM25Scorer
    from src.utils.initialize_lucene import initialize_lucene

    BM25_AVAILABLE = True
except ImportError as e:
    logging.warning(f"BM25 scorer components not available. BM25 scores will be zero. Error: {e}")
    TokenBM25Scorer = None
    initialize_lucene = None
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles the end-to-end evaluation of expansion models.
    """

    def __init__(self,
                 reranker: MultiVectorReranker,
                 rm_expansion: RMExpansion,
                 semantic_sim: SemanticSimilarity,
                 bm25_scorer: Optional[TokenBM25Scorer] = None,
                 top_k_pseudo_docs: int = 10):
        """Initializes the ModelEvaluator."""
        self.reranker = reranker
        self.rm_expansion = rm_expansion
        self.semantic_sim = semantic_sim
        self.bm25_scorer = bm25_scorer
        self.top_k_pseudo_docs = top_k_pseudo_docs
        logger.info("ModelEvaluator initialized.")

    def evaluate_model(self,
                       expansion_model: ExpansionModel,
                       eval_data: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Evaluates a single expansion model configuration on the test set.

        Args:
            expansion_model: The expansion model instance to evaluate.
            eval_data: A dictionary containing queries, documents, and first-stage runs.

        Returns:
            A dictionary representing the TREC-formatted run for the model.
        """
        queries = eval_data['queries']
        documents = eval_data['documents']
        first_stage_runs = eval_data['first_stage_runs']

        reranked_run = {}

        for qid, query_text in tqdm(queries.items(), desc=f"Evaluating model"):
            if qid not in first_stage_runs:
                continue

            # --- 1. Get pseudo-relevant documents ---
            top_docs = first_stage_runs[qid][:self.top_k_pseudo_docs]
            pseudo_docs_text = [documents.get(doc_id, "") for doc_id, _ in top_docs]
            pseudo_scores = [score for _, score in top_docs]
            reference_doc_id = top_docs[0][0] if top_docs else None

            if not any(pseudo_docs_text):
                reranked_run[qid] = [(doc_id, score) for doc_id, score in first_stage_runs[qid]]
                continue

            # --- 2. Get importance weights from the given model ---
            importance_weights = expansion_model.expand_query(
                query=query_text,
                pseudo_relevant_docs=pseudo_docs_text,
                pseudo_relevant_scores=pseudo_scores,
                reference_doc_id=reference_doc_id
            )

            # --- 3. Rerank candidates ---
            candidate_results = [(doc_id, documents.get(doc_id, ""), score) for doc_id, score in first_stage_runs[qid]]

            reranked_results = self.reranker.rerank(
                query=query_text,
                expansion_terms=[(term, 0) for term in importance_weights.keys()],
                importance_weights=importance_weights,
                candidate_results=candidate_results,
                top_k=1000  # Rerank all candidates
            )
            reranked_run[qid] = reranked_results

        return reranked_run


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained importance-weighted query expansion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Required Arguments ---
    parser.add_argument('--weights_file', type=str, required=True,
                        help='Path to the learned_weights.json file.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Name of the ir_datasets test set (e.g., "msmarco-passage/trec-dl-2020").')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results and run files.')

    # --- Model & Component Arguments ---
    parser.add_argument('--semantic_model', type=str, default='all-MiniLM-L6-v2',
                        help='Name of the sentence-transformer model for reranking.')

    # --- BM25 Arguments (Optional) ---
    parser.add_argument('--index_path', type=str, default=None,
                        help='Path to the pre-built BM25 index. Required for BM25-based models.')
    parser.add_argument('--lucene_path', type=str, default=None,
                        help='Path to Lucene JAR files. Required if --index_path is used.')

    # --- Evaluation Options ---
    parser.add_argument('--run_ablation', action='store_true',
                        help='If set, run a full ablation study with all baseline models.')
    parser.add_argument('--save_runs', action='store_true',
                        help='If set, save the TREC-formatted run file for each model.')

    # --- Logging Arguments ---
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    # --- Setup ---
    output_dir = ensure_dir(args.output_dir)
    runs_dir = ensure_dir(output_dir / 'runs') if args.save_runs else None
    logger = setup_experiment_logging("evaluate_model", args.log_level, str(output_dir / 'evaluation.log'))

    log_experiment_info(logger, **vars(args))

    try:
        # --- Load Learned Weights ---
        logger.info(f"Loading learned weights from: {args.weights_file}")
        learned_weights = load_learned_weights(args.weights_file)
        logger.info(
            f"Using weights: α={learned_weights[0]:.3f}, β={learned_weights[1]:.3f}, γ={learned_weights[2]:.3f}")

        # --- Initialize Components ---
        with TimedOperation(logger, "Initializing all components"):
            reranker = MultiVectorReranker(model_name=args.semantic_model)
            rm_expansion = RMExpansion()
            semantic_sim = SemanticSimilarity(model_name=args.semantic_model)

            bm25_scorer = None
            if args.index_path:
                if not BM25_AVAILABLE:
                    raise ImportError("BM25 components requested but not available.")
                if not args.lucene_path:
                    raise ValueError("--lucene_path is required when using --index_path.")
                initialize_lucene(args.lucene_path)
                bm25_scorer = TokenBM25Scorer(args.index_path)

        # --- Load Test Data ---
        with TimedOperation(logger, f"Loading test dataset: {args.dataset}"):
            dataset = ir_datasets.load(args.dataset)
            eval_data = {
                'queries': {q.query_id: q.text for q in dataset.queries_iter()},
                'qrels': {q.query_id: {d.doc_id: d.relevance for d in q.qrels_iter()} for q in dataset.queries_iter()},
                'documents': {d.doc_id: d.text for d in dataset.docs_iter()},
                'first_stage_runs': defaultdict(list)
            }
            if dataset.has_scoreddocs():
                for sdoc in dataset.scoreddocs_iter():
                    eval_data['first_stage_runs'][sdoc.query_id].append((sdoc.doc_id, sdoc.score))
                eval_data['first_stage_runs'] = dict(eval_data['first_stage_runs'])
        logger.info(f"Loaded {len(eval_data['queries'])} queries for evaluation.")

        # --- Instantiate the Evaluator ---
        evaluator = ModelEvaluator(reranker, rm_expansion, semantic_sim, bm25_scorer)

        # --- Define Models for Evaluation ---
        all_models_to_run = {}
        if args.run_ablation:
            logger.info("Creating baseline and ablation models for a full study.")
            all_models_to_run = create_baseline_comparison_models(
                rm_expansion, semantic_sim, bm25_scorer, learned_weights
            )
        else:
            logger.info("Evaluating only the final trained model.")
            # Create just the final model using the learned weights
            final_model = create_baseline_comparison_models(
                rm_expansion, semantic_sim, bm25_scorer, learned_weights
            )['our_method']
            all_models_to_run['our_method'] = final_model

        # --- Run Evaluation Loop ---
        all_runs = {}
        # Always include the original first-stage run as a baseline
        all_runs['FirstStage'] = eval_data['first_stage_runs']

        for model_name, expansion_model in all_models_to_run.items():
            logger.info(f"--- Evaluating model: {model_name} ---")
            with TimedOperation(logger, f"Evaluation for {model_name}"):
                run_results = evaluator.evaluate_model(expansion_model, eval_data)
                all_runs[model_name] = run_results

        # --- Compute Final Metrics and Save ---
        trec_evaluator = TRECEvaluator()
        comparison_results = trec_evaluator.compare_runs(all_runs, eval_data['qrels'], baseline_run='FirstStage')

        log_results(logger, comparison_results, "FINAL EVALUATION RESULTS")

        # Save results to disk
        save_json(comparison_results, output_dir / 'evaluation_metrics.json')

        if args.save_runs and runs_dir:
            logger.info(f"Saving all TREC run files to: {runs_dir}")
            for run_name, run_data in all_runs.items():
                save_trec_run(run_data, runs_dir / f"{run_name}.txt", run_name=run_name)

        # Print a clean summary table to the console
        print("\n" + "=" * 80)
        print(f"Final Results on {args.dataset}")
        print("=" * 80)
        print(trec_evaluator.create_results_table(comparison_results))
        print("=" * 80)

    except Exception as e:
        logger.critical(f"A critical error occurred during model evaluation: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)

    logger.info("EVALUATION SCRIPT COMPLETED SUCCESSFULLY")
    logger.info(f"All results saved in: {output_dir}")


if __name__ == "__main__":
    main()