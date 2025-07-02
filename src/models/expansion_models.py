"""
Expansion Models Module

Integrates all components to create complete query expansion models.
Provides different expansion strategies and the main importance-weighted approach.

Author: Your Name
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod

# Import your core modules
from src.core.rm_expansion import RMExpansion
from src.core.semantic_similarity import SemanticSimilarity
from src.core.bm25_scorer import TokenBM25Scorer

logger = logging.getLogger(__name__)


class ExpansionModel(ABC):
    """
    Abstract base class for query expansion models.
    """

    @abstractmethod
    def expand_query(self,
                     query: str,
                     pseudo_relevant_docs: List[str],
                     pseudo_relevant_scores: List[float],
                     reference_doc_id: Optional[str] = None) -> Dict[str, float]:
        """
        Expand query and return importance weights for expansion terms.

        Args:
            query: Original query text
            pseudo_relevant_docs: List of pseudo-relevant documents
            pseudo_relevant_scores: Relevance scores for documents
            reference_doc_id: Reference document ID for BM25 scoring

        Returns:
            Dictionary mapping expansion terms to importance weights
        """
        pass


class UniformExpansionModel(ExpansionModel):
    """
    Baseline expansion model that assigns uniform weights to all expansion terms.
    """

    def __init__(self,
                 rm_expansion: RMExpansion,
                 num_expansion_terms: int = 15):
        """
        Initialize uniform expansion model.

        Args:
            rm_expansion: RM expansion instance
            num_expansion_terms: Number of expansion terms to use
        """
        self.rm_expansion = rm_expansion
        self.num_expansion_terms = num_expansion_terms

        logger.info("Uniform expansion model initialized")

    def expand_query(self,
                     query: str,
                     pseudo_relevant_docs: List[str],
                     pseudo_relevant_scores: List[float],
                     reference_doc_id: Optional[str] = None) -> Dict[str, float]:
        """
        Expand query with uniform importance weights.

        Returns:
            Dictionary with uniform weights (1.0) for all expansion terms
        """
        # Get RM expansion terms
        rm_terms = self.rm_expansion.expand_query(
            query=query,
            documents=pseudo_relevant_docs,
            scores=pseudo_relevant_scores,
            num_expansion_terms=self.num_expansion_terms,
            rm_type="rm3"
        )

        # Assign uniform weights
        importance_weights = {}
        for term, rm_weight in rm_terms:
            importance_weights[term] = 1.0  # Uniform weight

        logger.debug(f"Uniform expansion: {len(importance_weights)} terms with weight 1.0")
        return importance_weights


class RMOnlyExpansionModel(ExpansionModel):
    """
    Expansion model that uses only RM weights (no BM25 or semantic similarity).
    """

    def __init__(self,
                 rm_expansion: RMExpansion,
                 num_expansion_terms: int = 15):
        """
        Initialize RM-only expansion model.

        Args:
            rm_expansion: RM expansion instance
            num_expansion_terms: Number of expansion terms to use
        """
        self.rm_expansion = rm_expansion
        self.num_expansion_terms = num_expansion_terms

        logger.info("RM-only expansion model initialized")

    def expand_query(self,
                     query: str,
                     pseudo_relevant_docs: List[str],
                     pseudo_relevant_scores: List[float],
                     reference_doc_id: Optional[str] = None) -> Dict[str, float]:
        """
        Expand query using only RM weights.

        Returns:
            Dictionary with RM weights as importance scores
        """
        # Get RM expansion terms
        rm_terms = self.rm_expansion.expand_query(
            query=query,
            documents=pseudo_relevant_docs,
            scores=pseudo_relevant_scores,
            num_expansion_terms=self.num_expansion_terms,
            rm_type="rm3"
        )

        # Use RM weights directly
        importance_weights = {}
        for term, rm_weight in rm_terms:
            importance_weights[term] = rm_weight

        logger.debug(f"RM-only expansion: {len(importance_weights)} terms")
        return importance_weights


class ImportanceWeightedExpansionModel(ExpansionModel):
    """
    Main contribution: Importance-weighted expansion model that combines
    RM weights, BM25 scores, and semantic similarity using learned weights.
    """

    def __init__(self,
                 rm_expansion: RMExpansion,
                 semantic_similarity: SemanticSimilarity,
                 bm25_scorer: Optional[TokenBM25Scorer] = None,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 num_expansion_terms: int = 15):
        """
        Initialize importance-weighted expansion model.

        Args:
            rm_expansion: RM expansion instance
            semantic_similarity: Semantic similarity computer
            bm25_scorer: BM25 scorer (optional, scores will be 0.0 if None)
            alpha: Weight for RM component
            beta: Weight for BM25 component
            gamma: Weight for semantic similarity component
            num_expansion_terms: Number of expansion terms to use
        """
        self.rm_expansion = rm_expansion
        self.semantic_similarity = semantic_similarity
        self.bm25_scorer = bm25_scorer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_expansion_terms = num_expansion_terms

        logger.info(f"Importance-weighted expansion model initialized")
        logger.info(f"Weights: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")
        logger.info(f"BM25 available: {bm25_scorer is not None}")

    def set_weights(self, alpha: float, beta: float, gamma: float):
        """
        Update the learned weights.

        Args:
            alpha: Weight for RM component
            beta: Weight for BM25 component
            gamma: Weight for semantic similarity component
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        logger.info(f"Updated weights: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")

    def expand_query(self,
                     query: str,
                     pseudo_relevant_docs: List[str],
                     pseudo_relevant_scores: List[float],
                     reference_doc_id: Optional[str] = None) -> Dict[str, float]:
        """
        Expand query using learned importance weighting.

        Returns:
            Dictionary with learned importance weights for expansion terms
        """
        # Step 1: Get RM expansion terms
        rm_terms = self.rm_expansion.expand_query(
            query=query,
            documents=pseudo_relevant_docs,
            scores=pseudo_relevant_scores,
            num_expansion_terms=self.num_expansion_terms,
            rm_type="rm3"
        )

        if not rm_terms:
            logger.warning("No RM expansion terms found")
            return {}

        # Step 2: Get expansion term list for batch processing
        expansion_term_list = [term for term, _ in rm_terms]

        # Step 3: Compute semantic similarities (batch)
        semantic_scores = self.semantic_similarity.compute_query_expansion_similarities(
            query, expansion_term_list
        )

        # Step 4: Compute BM25 scores if available
        if self.bm25_scorer:
            bm25_scores = self.bm25_scorer.compute_collection_level_bm25(expansion_term_list)
        else:
            bm25_scores = {term: 0.0 for term in expansion_term_list}

        # Step 5: Combine all scores using learned weights
        importance_weights = {}

        for term, rm_weight in rm_terms:
            # Get individual components
            rm_component = rm_weight
            bm25_component = bm25_scores.get(term, 0.0)
            semantic_component = semantic_scores.get(term, 0.0)

            # Combine using learned weights (your key contribution)
            importance_score = (self.alpha * rm_component +
                                self.beta * bm25_component +
                                self.gamma * semantic_component)

            importance_weights[term] = importance_score

        logger.debug(f"Importance-weighted expansion: {len(importance_weights)} terms")
        return importance_weights

    def explain_importance_computation(self,
                                       query: str,
                                       pseudo_relevant_docs: List[str],
                                       pseudo_relevant_scores: List[float],
                                       reference_doc_id: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Explain how importance scores are computed for each term.
        Useful for debugging and analysis.

        Returns:
            Dictionary with detailed breakdown for each term
        """
        # Get RM expansion terms
        rm_terms = self.rm_expansion.expand_query(
            query=query,
            documents=pseudo_relevant_docs,
            scores=pseudo_relevant_scores,
            num_expansion_terms=self.num_expansion_terms,
            rm_type="rm3"
        )

        if not rm_terms:
            return {}

        # Get all component scores
        expansion_term_list = [term for term, _ in rm_terms]
        semantic_scores = self.semantic_similarity.compute_query_expansion_similarities(
            query, expansion_term_list
        )

        if self.bm25_scorer and reference_doc_id:
            bm25_scores = self.bm25_scorer.compute_bm25_term_weight(
                reference_doc_id, expansion_term_list
            )
        else:
            bm25_scores = {term: 0.0 for term in expansion_term_list}

        # Create detailed breakdown
        explanation = {}

        for term, rm_weight in rm_terms:
            rm_component = rm_weight
            bm25_component = bm25_scores.get(term, 0.0)
            semantic_component = semantic_scores.get(term, 0.0)

            final_score = (self.alpha * rm_component +
                           self.beta * bm25_component +
                           self.gamma * semantic_component)

            explanation[term] = {
                'rm_weight': rm_component,
                'bm25_score': bm25_component,
                'semantic_score': semantic_component,
                'weighted_rm': self.alpha * rm_component,
                'weighted_bm25': self.beta * bm25_component,
                'weighted_semantic': self.gamma * semantic_component,
                'final_importance': final_score
            }

        return explanation


def create_expansion_model(model_type: str,
                           rm_expansion: RMExpansion,
                           semantic_similarity: SemanticSimilarity,
                           bm25_scorer: Optional[TokenBM25Scorer] = None,
                           **kwargs) -> ExpansionModel:
    """
    Factory function to create expansion models.

    Args:
        model_type: Type of expansion model ('uniform', 'rm_only', 'importance_weighted')
        rm_expansion: RM expansion instance
        semantic_similarity: Semantic similarity computer
        bm25_scorer: BM25 scorer (optional)
        **kwargs: Additional arguments for the model

    Returns:
        ExpansionModel instance

    Raises:
        ValueError: If unknown model type
    """
    if model_type.lower() == 'uniform':
        return UniformExpansionModel(rm_expansion, **kwargs)
    elif model_type.lower() == 'rm_only':
        return RMOnlyExpansionModel(rm_expansion, **kwargs)
    elif model_type.lower() == 'importance_weighted':
        return ImportanceWeightedExpansionModel(
            rm_expansion, semantic_similarity, bm25_scorer, **kwargs
        )
    else:
        raise ValueError(f"Unknown expansion model type: {model_type}")


def create_baseline_comparison_models(rm_expansion: RMExpansion,
                                      semantic_similarity: SemanticSimilarity,
                                      bm25_scorer: Optional[TokenBM25Scorer] = None,
                                      learned_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[
    str, ExpansionModel]:
    """
    Create a set of expansion models for baseline comparison.

    Args:
        rm_expansion: RM expansion instance
        semantic_similarity: Semantic similarity computer
        bm25_scorer: BM25 scorer (optional)
        learned_weights: Tuple of (alpha, beta, gamma) learned weights

    Returns:
        Dictionary mapping model names to expansion models
    """
    alpha, beta, gamma = learned_weights

    models = {
        'uniform': UniformExpansionModel(rm_expansion),
        'rm_only': RMOnlyExpansionModel(rm_expansion),
        'bm25_only': ImportanceWeightedExpansionModel(
            rm_expansion, semantic_similarity, bm25_scorer,
            alpha=0.0, beta=1.0, gamma=0.0
        ),
        'semantic_only': ImportanceWeightedExpansionModel(
            rm_expansion, semantic_similarity, bm25_scorer,
            alpha=0.0, beta=0.0, gamma=1.0
        ),
        'rm_bm25': ImportanceWeightedExpansionModel(
            rm_expansion, semantic_similarity, bm25_scorer,
            alpha=alpha, beta=beta, gamma=0.0
        ),
        'rm_semantic': ImportanceWeightedExpansionModel(
            rm_expansion, semantic_similarity, bm25_scorer,
            alpha=alpha, beta=0.0, gamma=gamma
        ),
        'bm25_semantic': ImportanceWeightedExpansionModel(
            rm_expansion, semantic_similarity, bm25_scorer,
            alpha=0.0, beta=beta, gamma=gamma
        ),
        'our_method': ImportanceWeightedExpansionModel(
            rm_expansion, semantic_similarity, bm25_scorer,
            alpha=alpha, beta=beta, gamma=gamma
        )
    }

    logger.info(f"Created {len(models)} expansion models for comparison")
    return models


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for example
    logging.basicConfig(level=logging.INFO)

    print("Expansion Models Module Example")
    print("=" * 40)

    # Initialize core components
    print("1. Initializing core components...")
    rm_expansion = RMExpansion()
    semantic_similarity = SemanticSimilarity('all-MiniLM-L6-v2')

    # Mock BM25 scorer (in practice, use real one)
    bm25_scorer = None  # Set to None for this example

    # Example data
    query = "machine learning algorithms"
    pseudo_relevant_docs = [
        "Neural networks are powerful machine learning models for classification.",
        "Supervised learning algorithms require labeled training data.",
        "Deep learning has revolutionized computer vision and NLP."
    ]
    pseudo_relevant_scores = [0.9, 0.8, 0.7]

    print(f"Query: {query}")
    print(f"Pseudo-relevant docs: {len(pseudo_relevant_docs)}")
    print()

    # Create different expansion models
    print("2. Creating expansion models...")

    # Uniform baseline
    uniform_model = create_expansion_model(
        'uniform', rm_expansion, semantic_similarity, bm25_scorer
    )

    # RM-only baseline
    rm_only_model = create_expansion_model(
        'rm_only', rm_expansion, semantic_similarity, bm25_scorer
    )

    # Our importance-weighted model (with example learned weights)
    our_model = create_expansion_model(
        'importance_weighted', rm_expansion, semantic_similarity, bm25_scorer,
        alpha=1.2, beta=0.8, gamma=1.5  # Example learned weights
    )

    print("Created models: uniform, rm_only, importance_weighted")
    print()

    # Compare expansion results
    print("3. Comparing expansion results:")

    models = {
        'Uniform': uniform_model,
        'RM Only': rm_only_model,
        'Our Method': our_model
    }

    for model_name, model in models.items():
        importance_weights = model.expand_query(
            query, pseudo_relevant_docs, pseudo_relevant_scores
        )

        print(f"\n{model_name} Expansion:")
        # Show top 5 terms
        sorted_terms = sorted(importance_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (term, weight) in enumerate(sorted_terms, 1):
            print(f"  {i}. {term:<15} {weight:.4f}")

    # Detailed explanation for our method
    print("\n4. Detailed importance computation (Our Method):")
    if isinstance(our_model, ImportanceWeightedExpansionModel):
        explanation = our_model.explain_importance_computation(
            query, pseudo_relevant_docs, pseudo_relevant_scores
        )

        print(f"Weights: alpha=1.2, beta=0.8, gamma=1.5")
        print("Term breakdown (top 3):")

        sorted_explanation = sorted(explanation.items(),
                                    key=lambda x: x[1]['final_importance'], reverse=True)[:3]

        for term, breakdown in sorted_explanation:
            print(f"\n  {term}:")
            print(f"    RM: {breakdown['rm_weight']:.3f} * 1.2 = {breakdown['weighted_rm']:.3f}")
            print(f"    BM25: {breakdown['bm25_score']:.3f} * 0.8 = {breakdown['weighted_bm25']:.3f}")
            print(f"    Semantic: {breakdown['semantic_score']:.3f} * 1.5 = {breakdown['weighted_semantic']:.3f}")
            print(f"    Final: {breakdown['final_importance']:.3f}")

    print("\n5. Creating baseline comparison set:")
    baseline_models = create_baseline_comparison_models(
        rm_expansion, semantic_similarity, bm25_scorer,
        learned_weights=(1.2, 0.8, 1.5)
    )

    print(f"Created {len(baseline_models)} models for ablation study:")
    for model_name in baseline_models.keys():
        print(f"  - {model_name}")

    print("\nExpansion models ready for evaluation!")
    print("Use these models with your TREC DL reranking pipeline.")