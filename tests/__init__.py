"""
Test suite for expansion weight learning package.

This module contains comprehensive unit tests for all components:
- Core functionality tests (RM expansion, semantic similarity)
- Model tests (weight optimizers, multi-vector retrieval)
- Evaluation tests (metrics, evaluators)
- Data creation and processing tests
- Integration tests

Test modules:
- test_rm_expansion: Tests for RM expansion algorithms
- test_weight_optimizer: Tests for weight optimization algorithms
- test_data_creator: Tests for training data creation
- test_evaluation: Tests for evaluation functionality

Usage:
    # Run all tests
    python -m pytest tests/ -v

    # Run specific test module
    python -m pytest tests/test_rm_expansion.py -v

    # Run tests with coverage
    python -m pytest tests/ --cov=src --cov-report=html
"""

import sys
import unittest
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """
    Run all tests in the test suite.

    Returns:
        bool: True if all tests pass, False otherwise
    """
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).parent), pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def run_test_module(module_name):
    """
    Run tests from a specific module.

    Args:
        module_name (str): Name of the test module (e.g., 'test_rm_expansion')

    Returns:
        bool: True if tests pass, False otherwise
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(module_name)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


# Test configuration
TEST_CONFIG = {
    'verbose': True,
    'failfast': False,
    'buffer': True,
    'catch': True
}

# Available test modules
TEST_MODULES = [
    'test_rm_expansion',
    'test_weight_optimizer',
    'test_data_creator',
    'test_evaluation'
]

__all__ = [
    'run_all_tests',
    'run_test_module',
    'TEST_CONFIG',
    'TEST_MODULES'
]