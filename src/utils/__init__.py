"""
Utility functions for expansion weight learning.

This module provides essential utilities:
- Logging configuration and experiment tracking
- File I/O operations for training data, models, and results
- Lucene initialization utilities for BM25 integration

Key modules:
- logging_utils: Comprehensive logging setup and experiment tracking
- file_utils: File operations with proper format handling
- lucene_utils: Lucene JVM initialization and management
"""

from .logging_utils import (
    setup_logging,
    setup_experiment_logging,
    get_logger,
    log_experiment_info,
    log_results,
    TimedOperation
)

from .file_utils import (
    ensure_dir,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    save_trec_run,
    load_trec_run,
    save_qrels,
    load_qrels,
    save_training_data,
    load_training_data,
    save_experiment_results,
    save_learned_weights,
    load_learned_weights,
    TemporaryDirectory
)

# Optional Lucene utilities (might not be available in all environments)
try:
    from .lucene_utils import initialize_lucene, check_lucene_availability

    LUCENE_AVAILABLE = True
    __all_lucene = ['initialize_lucene', 'check_lucene_availability']
except ImportError:
    LUCENE_AVAILABLE = False
    __all_lucene = []

__all__ = [
              # Logging utilities
              'setup_logging',
              'setup_experiment_logging',
              'get_logger',
              'log_experiment_info',
              'log_results',
              'TimedOperation',

              # File utilities
              'ensure_dir',
              'save_json',
              'load_json',
              'save_pickle',
              'load_pickle',
              'save_trec_run',
              'load_trec_run',
              'save_qrels',
              'load_qrels',
              'save_training_data',
              'load_training_data',
              'save_experiment_results',
              'save_learned_weights',
              'load_learned_weights',
              'TemporaryDirectory'
          ] + __all_lucene