"""
Expansion Weight Learning Package

A modular framework for learning importance weights in query expansion
for multi-vector dense retrieval.

This package provides:
- RM expansion with importance weighting
- BM25 and semantic similarity integration
- Multi-vector retrieval with learned weights
- Comprehensive evaluation tools

Main modules:
- core: Core functionality (RM expansion, BM25, semantic similarity)
- models: Model implementations (weight optimizers, multi-vector retrieval)
- evaluation: Evaluation metrics and tools
- utils: Utility functions (logging, file I/O)
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes for convenient access
try:
    from .core.rm_expansion import RMExpansion, rm1_expansion, rm3_expansion
    from .core.semantic_similarity import SemanticSimilarity
    from .models.weight_optimizer import LBFGSOptimizer, GridSearchOptimizer, create_optimizer
    from .evaluation.evaluator import TRECEvaluator, ExpansionEvaluator, create_trec_dl_evaluator
except ImportError:
    # Graceful degradation if dependencies are not available
    pass

# Package metadata
__all__ = [
    # Core classes
    'RMExpansion',
    'SemanticSimilarity',

    # Model classes
    'LBFGSOptimizer',
    'GridSearchOptimizer',
    'create_optimizer',

    # Evaluation classes
    'TRECEvaluator',
    'ExpansionEvaluator',
    'create_trec_dl_evaluator',

    # Convenience functions
    'rm1_expansion',
    'rm3_expansion',
]