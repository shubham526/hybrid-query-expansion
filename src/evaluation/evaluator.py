import logging
import tempfile
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import pytrec_eval
import ir_datasets
from .metrics import get_metric

logger = logging.getLogger(__name__)


class TRECEvaluator:
    """
    Evaluator for TREC-style retrieval experiments.
    Handles qrels, runs, and computes standard IR metrics.
    """

    def __init__(self, metrics: List[str] = None):
        """
        Initialize evaluator with specified metrics.

        Args:
            metrics: List of metrics to compute (e.g., ['map', 'ndcg_cut_10', 'recip_rank'])
        """
        if metrics is None:
            self.metrics = ['map', 'ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_10', 'recall_100']
        else:
            self.metrics = metrics

        logger.info(f"TRECEvaluator initialized with metrics: {self.metrics}")

    def evaluate_run(self, run_results: Dict[str, List[Tuple[str, float]]],
                     qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """
        Evaluate a single run against qrels.

        Args:
            run_results: {query_id: [(doc_id, score), ...]}
            qrels: {query_id: {doc_id: relevance}}

        Returns:
            Dictionary of metric scores
        """
        # Create temporary files for pytrec_eval
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.qrel') as qrel_file, \
                tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.run') as run_file:

            # Write qrels
            self._write_qrels(qrels, qrel_file.name)

            # Write run
            self._write_run(run_results, run_file.name)

            # Evaluate using your existing get_metric function
            results = {}
            for metric in self.metrics:
                try:
                    results[metric] = get_metric(qrel_file.name, run_file.name, metric)
                except Exception as e:
                    logger.warning(f"Failed to compute {metric}: {e}")
                    results[metric] = 0.0

            # Cleanup
            os.unlink(qrel_file.name)
            os.unlink(run_file.name)

            return results

    def evaluate_multiple_runs(self, runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
                               qrels: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple runs against qrels.

        Args:
            runs: {run_name: {query_id: [(doc_id, score), ...]}}
            qrels: {query_id: {doc_id: relevance}}

        Returns:
            {run_name: {metric: score}}
        """
        logger.info(f"Evaluating {len(runs)} runs on {len(qrels)} queries")

        results = {}
        for run_name, run_results in runs.items():
            logger.info(f"Evaluating run: {run_name}")
            results[run_name] = self.evaluate_run(run_results, qrels)

        return results

    def compare_runs(self, runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
                     qrels: Dict[str, Dict[str, int]],
                     baseline_run: str = None) -> Dict[str, Any]:
        """
        Compare multiple runs and compute improvements over baseline.

        Args:
            runs: {run_name: {query_id: [(doc_id, score), ...]}}
            qrels: {query_id: {doc_id: relevance}}
            baseline_run: Name of baseline run for computing improvements

        Returns:
            Comprehensive comparison results
        """
        # Evaluate all runs
        evaluations = self.evaluate_multiple_runs(runs, qrels)

        # Find baseline
        if baseline_run is None:
            baseline_run = list(runs.keys())[0]
            logger.info(f"Using '{baseline_run}' as baseline")

        baseline_scores = evaluations[baseline_run]

        # Compute improvements
        comparison = {
            'evaluations': evaluations,
            'baseline': baseline_run,
            'improvements': {}
        }

        for run_name, scores in evaluations.items():
            if run_name == baseline_run:
                continue

            improvements = {}
            for metric, score in scores.items():
                baseline_score = baseline_scores[metric]
                if baseline_score > 0:
                    improvement = (score - baseline_score) / baseline_score * 100
                    improvements[f"{metric}_improvement_pct"] = improvement
                    improvements[f"{metric}_improvement_abs"] = score - baseline_score
                else:
                    improvements[f"{metric}_improvement_pct"] = 0.0
                    improvements[f"{metric}_improvement_abs"] = score

            comparison['improvements'][run_name] = improvements

        return comparison

    def create_results_table(self, comparison_results: Dict[str, Any]) -> str:
        """
        Create a formatted results table for paper inclusion.
        """
        evaluations = comparison_results['evaluations']
        baseline = comparison_results['baseline']
        improvements = comparison_results['improvements']

        # Table header
        table = "Method" + "\t" + "\t".join(self.metrics) + "\n"
        table += "-" * (len("Method") + sum(len(m) + 1 for m in self.metrics)) + "\n"

        # Baseline row
        baseline_scores = evaluations[baseline]
        table += f"{baseline}"
        for metric in self.metrics:
            table += f"\t{baseline_scores[metric]:.4f}"
        table += "\n"

        # Other runs with improvements
        for run_name, scores in evaluations.items():
            if run_name == baseline:
                continue

            table += f"{run_name}"
            for metric in self.metrics:
                score = scores[metric]
                improvement = improvements[run_name].get(f"{metric}_improvement_abs", 0.0)
                if improvement > 0:
                    table += f"\t{score:.4f} (+{improvement:.4f})"
                else:
                    table += f"\t{score:.4f} ({improvement:.4f})"
            table += "\n"

        return table

    def _write_qrels(self, qrels: Dict[str, Dict[str, int]], filename: str):
        """Write qrels to TREC format file."""
        with open(filename, 'w') as f:
            for query_id, docs in qrels.items():
                for doc_id, relevance in docs.items():
                    f.write(f"{query_id} 0 {doc_id} {relevance}\n")

    def _write_run(self, run_results: Dict[str, List[Tuple[str, float]]], filename: str):
        """Write run results to TREC format file."""
        with open(filename, 'w') as f:
            for query_id, docs in run_results.items():
                for rank, (doc_id, score) in enumerate(docs, 1):
                    f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} run\n")


class ExpansionEvaluator(TRECEvaluator):
    """
    Specialized evaluator for query expansion experiments.
    """

    def __init__(self, metrics: List[str] = None):
        super().__init__(metrics)

    def evaluate_expansion_ablation(self,
                                    baseline_run: Dict[str, List[Tuple[str, float]]],
                                    expansion_models: Dict[str, Any],
                                    queries: Dict[str, str],
                                    qrels: Dict[str, Dict[str, int]],
                                    first_stage_runs: Dict[str, List[Tuple[str, float]]],
                                    reranker) -> Dict[str, Any]:
        """
        Evaluate ablation study for different expansion models.

        Args:
            baseline_run: Results without expansion
            expansion_models: {model_name: expansion_model}
            queries: {query_id: query_text}
            qrels: {query_id: {doc_id: relevance}}
            first_stage_runs: {query_id: [(doc_id, score), ...]}
            reranker: MultiVectorReranker instance

        Returns:
            Comprehensive ablation results
        """
        logger.info(f"Running expansion ablation with {len(expansion_models)} models")

        # Prepare runs dictionary
        runs = {'Baseline (No Expansion)': baseline_run}

        # Generate expansion terms for all queries (you may want to pre-compute these)
        expansion_terms_dict = self._generate_expansion_terms(queries)

        # Evaluate each expansion model
        for model_name, expansion_model in expansion_models.items():
            logger.info(f"Evaluating expansion model: {model_name}")

            # Compute importance weights for all queries
            importance_weights_dict = {}
            for query_id, query_text in queries.items():
                try:
                    # You'll need to provide pseudo-relevant docs and scores here
                    # This is a simplified version - adapt based on your RM implementation
                    importance_weights = expansion_model.expand_query(
                        query=query_text,
                        pseudo_relevant_docs=[],  # You'll need to provide these
                        pseudo_relevant_scores=[],  # You'll need to provide these
                        reference_doc_id=None  # For BM25 scoring
                    )
                    importance_weights_dict[query_id] = importance_weights
                except Exception as e:
                    logger.warning(f"Failed to compute importance for query {query_id}: {e}")
                    importance_weights_dict[query_id] = {}

            # Rerank using this expansion model
            try:
                reranked_results = reranker.rerank_trec_dl_run(
                    queries=queries,
                    first_stage_runs=first_stage_runs,
                    expansion_terms_dict=expansion_terms_dict,
                    importance_weights_dict=importance_weights_dict,
                    top_k=100
                )
                runs[model_name] = reranked_results
            except Exception as e:
                logger.error(f"Failed to rerank with model {model_name}: {e}")
                runs[model_name] = baseline_run  # Fallback to baseline

        # Compare all runs
        comparison = self.compare_runs(runs, qrels, 'Baseline (No Expansion)')

        return comparison

    def _generate_expansion_terms(self, queries: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate expansion terms for all queries.
        This is a placeholder - you'll need to implement based on your RM expansion.
        """
        # Placeholder implementation
        expansion_terms_dict = {}
        for query_id, query_text in queries.items():
            # You'll need to implement actual RM expansion here
            # This is just a placeholder
            terms = query_text.split()[:5]  # Use first 5 terms as "expansion"
            expansion_terms_dict[query_id] = [(term, 1.0) for term in terms]

        return expansion_terms_dict


def create_trec_dl_evaluator(year: str = "2019") -> ExpansionEvaluator:
    """
    Factory function to create evaluator for TREC DL datasets.

    Args:
        year: "2019" or "2020"

    Returns:
        Configured ExpansionEvaluator
    """
    # TREC DL specific metrics
    trec_dl_metrics = ['map', 'ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_100']

    evaluator = ExpansionEvaluator(metrics=trec_dl_metrics)

    logger.info(f"Created TREC DL {year} evaluator")
    return evaluator


# Example usage for your paper
def run_paper_evaluation():
    """
    Example evaluation pipeline for your SIGIR paper.
    """
    from models.multivector_retrieval import TRECDLReranker
    from expansion_models import create_baseline_comparison_models

    # Load TREC DL data
    evaluator = create_trec_dl_evaluator("2019")

    # Load data using ir_datasets
    dataset = ir_datasets.load("msmarco-passage/trec-dl-2019")
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    qrels = defaultdict(dict)
    for qrel in dataset.qrels_iter():
        qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

    # Load first-stage runs
    first_stage_runs = {}
    for scoreddoc in dataset.scoreddocs_iter():
        if scoreddoc.query_id not in first_stage_runs:
            first_stage_runs[scoreddoc.query_id] = []
        first_stage_runs[scoreddoc.query_id].append((scoreddoc.doc_id, scoreddoc.score))

    # Create expansion models (your baselines)
    expansion_models = create_baseline_comparison_models(
        rm_expansion=None,  # Your RM expansion instance
        semantic_sim=None,  # Your semantic similarity instance
        bm25_scorer=None,  # Your BM25 scorer instance
        learned_weights=(1.2, 0.8, 1.5)  # Your learned weights
    )

    # Run evaluation
    results = evaluator.evaluate_expansion_ablation(
        baseline_run=first_stage_runs,
        expansion_models=expansion_models,
        queries=queries,
        qrels=dict(qrels),
        first_stage_runs=first_stage_runs,
        reranker=None  # Your reranker instance
    )

    # Print results table
    print(evaluator.create_results_table(results))

    return results