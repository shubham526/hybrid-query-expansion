"""
Model implementations for expansion weight learning.

This module contains the core model implementations:
- Weight optimization algorithms (L-BFGS-B, Grid Search, Random Search)
- Multi-vector retrieval systems with importance weighting
- Expansion model variants for ablation studies

Key classes:
- WeightOptimizer: Base class for weight optimization
- LBFGSOptimizer: L-BFGS-B optimization for weight learning
- GridSearchOptimizer: Grid search optimization
- RandomSearchOptimizer: Random search optimization
- MultiVectorRetrieval: Multi-vector retrieval with importance weighting
"""

from .weight_optimizer import (
    WeightOptimizer,
    LBFGSOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    create_optimizer
)

try:
    from .multivector_retrieval import (
        MultiVectorRetrieval,
        TRECDLReranker,
        create_trec_dl_evaluation_pipeline
    )
    MULTIVECTOR_AVAILABLE = True
except ImportError:
    # Multi-vector retrieval might require additional dependencies
    MULTIVECTOR_AVAILABLE = False

try:
    from .expansion_models import (
        ExpansionModel,
        UniformExpansionModel,
        RMOnlyExpansionModel,
        ImportanceWeightedExpansionModel,
        create_baseline_comparison_models
    )
    EXPANSION_MODELS_AVAILABLE = True
except ImportError:
    # Expansion models might require additional dependencies
    EXPANSION_MODELS_AVAILABLE = False

# Export available classes
__all__ = [
    'WeightOptimizer',
    'LBFGSOptimizer',
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'create_optimizer'
]

if MULTIVECTOR_AVAILABLE:
    __all__.extend([
        'MultiVectorRetrieval',
        'TRECDLReranker',
        'create_trec_dl_evaluation_pipeline'
    ])

if EXPANSION_MODELS_AVAILABLE:
    __all__.extend([
        'ExpansionModel',
        'UniformExpansionModel',
        'RMOnlyExpansionModel',
        'ImportanceWeightedExpansionModel',
        'create_baseline_comparison_models'
    ])