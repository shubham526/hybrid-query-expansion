#!/usr/bin/env python3
"""
Evaluates a trained importance-weighted query expansion model on a test set.

This script performs the final evaluation by:
1.  Loading learned weights (alpha, beta, gamma) from a file.
2.  Loading an initial candidate run from the best available source (scoreddocs or a run file).
3.  Running the full expansion and multi-vector reranking pipeline on the candidate set.
4.  Optionally conducting a comprehensive ablation study.
5.  Saving final run files and evaluation metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
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
from src.models.memory_efficient_reranker import create_memory_efficient_reranker
from src.evaluation.evaluator import TRECEvaluator
from src.utils.file_utils import load_learned_weights, load_trec_run, save_json, save_trec_run, ensure_dir
from src.utils.logging_utils import setup_experiment_logging, log_experiment_info, TimedOperation, log_results

# Conditionally import BM25 infrastructure
try:
    from src.core.bm25_scorer import TokenBM25Scorer
    from src.utils.lucene_utils import initialize_lucene

    BM25_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import from 'src'. Please run 'pip install -e .' from the project root. Error: {e}")
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



class ModelEvaluator:
    """Handles the end-to-end evaluation of expansion models."""

    def __init__(self,
                 reranker,
                 rm_expansion: RMExpansion,
                 semantic_sim: SemanticSimilarity,
                 bm25_scorer: Optional[TokenBM25Scorer] = None,
                 top_k_pseudo_docs: int = 10):
        self.reranker = reranker
        self.rm_expansion = rm_expansion
        self.semantic_sim = semantic_sim
        self.bm25_scorer = bm25_scorer
        self.top_k_pseudo_docs = top_k_pseudo_docs
        logger.info("ModelEvaluator initialized.")

    def evaluate_model(self,
                       expansion_model: ExpansionModel,
                       eval_data: Dict[str, Any]) -> Dict[str, List[tuple]]:
        """Evaluates a single expansion model configuration on the test set."""
        queries = eval_data['queries']
        documents = eval_data['documents']
        first_stage_runs = eval_data['first_stage_runs']

        reranked_run = {}

        for qid, query_text in tqdm(queries.items(), desc=f"Evaluating model"):
            if qid not in first_stage_runs:
                reranked_run[qid] = []
                continue

            top_docs_for_prf = first_stage_runs[qid][:self.top_k_pseudo_docs]
            pseudo_docs_text = [documents.get(doc_id, "") for doc_id, _ in top_docs_for_prf]
            pseudo_scores = [score for _, score in top_docs_for_prf]
            reference_doc_id = top_docs_for_prf[0][0] if top_docs_for_prf else None

            if not any(pseudo_docs_text):
                reranked_run[qid] = [(doc_id, score) for doc_id, score in first_stage_runs[qid]]
                continue

            importance_weights = expansion_model.expand_query(
                query=query_text,
                pseudo_relevant_docs=pseudo_docs_text,
                pseudo_relevant_scores=pseudo_scores,
                reference_doc_id=reference_doc_id
            )
            rm_terms = self.rm_expansion.expand_query(
                query=query_text,
                documents=pseudo_docs_text,
                scores=pseudo_scores,
                num_expansion_terms=20,
                rm_type="rm3"
            )
            rm_terms_dict = dict(rm_terms)

            candidate_results = [(doc_id, documents.get(doc_id, ""), score) for doc_id, score in first_stage_runs[qid]]

            reranked_results = self.reranker.rerank_streaming(
                query=query_text,
                expansion_terms=[(term, rm_terms_dict.get(term, 0.0)) for term in importance_weights.keys()],
                importance_weights=importance_weights,
                candidate_results=candidate_results,
                top_k=1000
            )
            reranked_run[qid] = reranked_results

        return reranked_run


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained importance-weighted query expansion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--weights-file', type=str, required=True, help='Path to the learned_weights.json file.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the ir_datasets test set.')
    parser.add_argument('--run-file-path', type=str,
                        help='Optional path to a baseline TREC run file for candidate generation.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save evaluation results and run files.')
    parser.add_argument('--query-ids-file', type=str, default=None,
                        help='Optional path to a file with query IDs to evaluate.')
    parser.add_argument('--semantic-model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformer model for reranking.')
    parser.add_argument('--index-path', type=str, default=None, help='Path to the pre-built BM25 index.')
    parser.add_argument('--lucene-path', type=str, default=None, help='Path to Lucene JAR files.')
    parser.add_argument('--run-ablation', action='store_true', help='If set, run a full ablation study.')
    parser.add_argument('--save-runs', action='store_true',
                        help='If set, save the TREC-formatted run file for each model.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    runs_dir = ensure_dir(output_dir / 'runs') if args.save_runs else None
    logger = setup_experiment_logging("evaluate_model", args.log_level, str(output_dir / 'evaluation.log'))
    log_experiment_info(logger, **vars(args))

    try:
        logger.info(f"Loading learned weights from: {args.weights_file}")
        learned_weights = load_learned_weights(args.weights_file)

        # STEP 1: Initialize basic components (no reranker yet)
        with TimedOperation(logger, "Initializing core components"):
            semantic_sim = SemanticSimilarity(model_name=args.semantic_model)
            bm25_scorer = None
            if args.index_path:
                if not BM25_AVAILABLE:
                    raise ImportError("BM25 components requested but not available.")
                if not args.lucene_path:
                    raise ValueError("--lucene-path is required for BM25.")
                initialize_lucene(args.lucene_path)
                bm25_scorer = TokenBM25Scorer(args.index_path)

        # STEP 2: Load dataset and build all_runs
        with TimedOperation(logger, f"Loading and filtering dataset: {args.dataset}"):
            dataset = ir_datasets.load(args.dataset)
            all_queries = {q.query_id: get_query_text(q) for q in dataset.queries_iter()}
            all_qrels = {qrel.query_id: {qrel.doc_id: qrel.relevance} for qrel in dataset.qrels_iter()}
            all_documents = {d.doc_id: (d.text if hasattr(d, 'text') else d.body) for d in dataset.docs_iter()}

            # Load candidate set (this builds all_runs)
            all_runs = defaultdict(list)
            if dataset.has_scoreddocs():
                logger.info("Found 'scoreddocs' in dataset. Using them as the candidate set.")
                for sdoc in dataset.scoreddocs_iter():
                    all_runs[sdoc.query_id].append((sdoc.doc_id, sdoc.score))
            elif args.run_file_path and Path(args.run_file_path).exists():
                logger.info(f"Using user-provided run file at '{args.run_file_path}' as the candidate set.")
                all_runs.update(load_trec_run(args.run_file_path))
            else:
                raise ValueError(
                    "A source for candidate documents is required for evaluation. Provide a run file via --run-file-path or use a dataset with scoreddocs.")

            # Filter queries if needed
            qids_to_evaluate = set(all_queries.keys())
            if args.query_ids_file:
                logger.info(f"Filtering evaluation to subset from: {args.query_ids_file}")
                with open(args.query_ids_file, 'r') as f:
                    qids_to_evaluate = {line.strip() for line in f if line.strip()}

            # Create evaluation data
            eval_data = {
                'queries': {qid: text for qid, text in all_queries.items() if qid in qids_to_evaluate},
                'qrels': {qid: qrels for qid, qrels in all_qrels.items() if qid in qids_to_evaluate},
                'documents': all_documents,
                'first_stage_runs': {qid: run for qid, run in all_runs.items() if qid in qids_to_evaluate}
            }

        logger.info(f"Loaded {len(eval_data['queries'])} queries for final evaluation.")

        # STEP 3: NOW initialize memory-efficient reranker (after all_runs is defined)
        with TimedOperation(logger, "Initializing memory-efficient reranker"):
            # Calculate total candidates based on actual data
            total_candidates = sum(len(candidates) for candidates in all_runs.values())
            large_candidate_sets = total_candidates > 50000
            logger.info(f"Total candidates across all queries: {total_candidates:,}")
            logger.info(f"Using {'large' if large_candidate_sets else 'standard'} candidate set optimization")

            reranker = create_memory_efficient_reranker(
                model_name=args.semantic_model,
                large_candidate_sets=large_candidate_sets
            )

        # STEP 4: Create evaluator and run evaluation
        evaluator = ModelEvaluator(reranker, semantic_sim, bm25_scorer)

        models_to_run = {}
        if args.run_ablation:
            logger.info("Creating all baseline and ablation models for a full study.")
            models_to_run = create_baseline_comparison_models(args.index_path, semantic_sim, bm25_scorer,
                                                              learned_weights)
        else:
            logger.info("Evaluating only the final trained model ('our_method').")
            models_to_run['our_method'] = \
                create_baseline_comparison_models(args.index_path, semantic_sim, bm25_scorer, learned_weights)[
                    'our_method']

        all_runs_for_eval = {'FirstStage': eval_data['first_stage_runs']}
        for model_name, expansion_model in models_to_run.items():
            logger.info(f"--- Evaluating model: {model_name} ---")
            with TimedOperation(logger, f"Evaluation for {model_name}"):
                run_results = evaluator.evaluate_model(expansion_model, eval_data)
                all_runs_for_eval[model_name] = run_results

        # STEP 5: Evaluate and save results
        trec_evaluator = TRECEvaluator()
        comparison_results = trec_evaluator.compare_runs(all_runs_for_eval, eval_data['qrels'],
                                                         baseline_run='FirstStage')
        log_results(logger, comparison_results, "FINAL EVALUATION RESULTS")
        save_json(comparison_results, output_dir / 'evaluation_metrics.json')

        if args.save_runs and runs_dir:
            logger.info(f"Saving all TREC run files to: {runs_dir}")
            for run_name, run_data in all_runs_for_eval.items():
                save_trec_run(run_data, runs_dir / f"{run_name}.txt", run_name=run_name)

        print("\n" + "=" * 80 + f"\nFinal Results on {args.dataset}" + (
            " (filtered)" if args.query_ids_file else "") + "\n" + "=" * 80)
        print(trec_evaluator.create_results_table(comparison_results))

        # Log final memory usage and cleanup
        if hasattr(reranker, 'get_memory_stats'):
            final_memory_stats = reranker.get_memory_stats()
            logger.info(f"Final memory statistics: {final_memory_stats}")
            # Clear caches before exit
            reranker.clear_caches()

        print("=" * 80)

    except Exception as e:
        logger.critical(f"A critical error occurred during model evaluation: {e}", exc_info=True)
        sys.exit(1)

    logger.info("EVALUATION SCRIPT COMPLETED SUCCESSFULLY")
    logger.info(f"All results saved in: {output_dir}")


if __name__ == "__main__":
    main()