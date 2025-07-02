"""
Core functionality for expansion weight learning.

This module provides the fundamental building blocks:
- RM expansion (RM1/RM3) algorithms
- Semantic similarity computation using sentence transformers
- Integration utilities for combining different importance signals

Key classes:
- RMExpansion: Relevance Model query expansion
- SemanticSimilarity: Sentence transformer-based similarity computation
"""

from .rm_expansion import RMExpansion, rm1_expansion, rm3_expansion
from .semantic_similarity import SemanticSimilarity

# Optional BM25 scorer import (might not be available in all environments)
try:
    from .bm25_scorer import TokenBM25Scorer
    __all__ = ['RMExpansion', 'SemanticSimilarity', 'TokenBM25Scorer', 'rm1_expansion', 'rm3_expansion']
except ImportError:
    # BM25 scorer not available (e.g., missing Java dependencies)
    __all__ = ['RMExpansion', 'SemanticSimilarity', 'rm1_expansion', 'rm3_expansion']