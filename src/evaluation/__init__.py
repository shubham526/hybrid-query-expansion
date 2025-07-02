"""
Evaluation tools for expansion weight learning.

This module provides comprehensive evaluation functionality:
- Standard IR metrics computation (nDCG, MAP, MRR, etc.)
- TREC-style evaluation with qrels and run files
- Expansion-specific evaluation and ablation studies
- Results comparison and statistical analysis

Key classes:
- TRECEvaluator: General TREC-style evaluation
- ExpansionEvaluator: Expansion-specific evaluation
"""

from .metrics import get_metric
from .evaluator import TRECEvaluator, ExpansionEvaluator, create_trec_dl_evaluator

__all__ = [
    'get_metric',
    'TRECEvaluator',
    'ExpansionEvaluator',
    'create_trec_dl_evaluator'
]