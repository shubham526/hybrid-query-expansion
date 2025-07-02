#!/usr/bin/env python3
"""
Unit tests for weight optimizer module.

Tests the optimization algorithms used for learning importance weights:
- L-BFGS-B optimizer
- Grid search optimizer
- Random search optimizer
- Evaluation function integration
- Convergence behavior
- Edge cases and error handling

Usage:
    python -m pytest test_weight_optimizer.py -v
    python test_weight_optimizer.py  # Run directly
"""

import unittest
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import tempfile
import json
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.weight_optimizer import (
    WeightOptimizer, LBFGSOptimizer, GridSearchOptimizer,
    RandomSearchOptimizer, create_optimizer
)


class TestWeightOptimizerBase(unittest.TestCase):
    """Test cases for base WeightOptimizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock training data
        self.mock_training_data = {
            'features': {
                'query1': {
                    'term_features': {
                        'neural': {'rm_weight': 0.8, 'bm25_score': 2.1, 'semantic_score': 0.7},
                        'networks': {'rm_weight': 0.6, 'bm25_score': 1.8, 'semantic_score': 0.6},
                        'algorithm': {'rm_weight': 0.4, 'bm25_score': 1.2, 'semantic_score': 0.5}
                    }
                },
                'query2': {
                    'term_features': {
                        'learning': {'rm_weight': 0.9, 'bm25_score': 2.5, 'semantic_score': 0.8},
                        'machine': {'rm_weight': 0.7, 'bm25_score': 2.0, 'semantic_score': 0.7},
                        'model': {'rm_weight': 0.5, 'bm25_score': 1.5, 'semantic_score': 0.6}
                    }
                }
            }
        }

        # Mock validation data
        self.mock_validation_queries = {
            'query1': 'neural networks algorithm',
            'query2': 'machine learning model'
        }

        self.mock_validation_qrels = {
            'query1': {'doc1': 2, 'doc2': 1, 'doc3': 0},
            'query2': {'doc1': 1, 'doc2': 2, 'doc3': 1}
        }

    def create_mock_evaluation_function(self, optimal_weights: Tuple[float, float, float] = (1.2, 0.8, 1.5)):
        """
        Create a mock evaluation function that has a known optimum.

        Args:
            optimal_weights: The weights that should yield maximum performance

        Returns:
            Mock evaluation function
        """

        def mock_eval_function(weights: Tuple[float, float, float]) -> float:
            """
            Mock evaluation function with known optimum.
            Uses quadratic distance from optimal weights.
            """
            alpha, beta, gamma = weights
            opt_alpha, opt_beta, opt_gamma = optimal_weights

            # Quadratic function with maximum at optimal_weights
            # f(w) = 1.0 - distance_penalty
            distance = ((alpha - opt_alpha) ** 2 +
                        (beta - opt_beta) ** 2 +
                        (gamma - opt_gamma) ** 2)

            # Scale to reasonable range (0.3 to 1.0)
            score = 1.0 - 0.7 * (distance / 10.0)
            return max(0.0, score)  # Ensure non-negative

        return mock_eval_function

    def create_noisy_evaluation_function(self, noise_level: float = 0.05):
        """Create evaluation function with noise to test robustness."""
        base_function = self.create_mock_evaluation_function()

        def noisy_eval_function(weights: Tuple[float, float, float]) -> float:
            base_score = base_function(weights)
            noise = np.random.normal(0, noise_level)
            return max(0.0, base_score + noise)

        return noisy_eval_function


class TestLBFGSOptimizer(TestWeightOptimizerBase):
    """Test cases for L-BFGS-B optimizer."""

    def setUp(self):
        """Set up L-BFGS optimizer tests."""
        super().setUp()
        self.optimizer = LBFGSOptimizer()

    def test_initialization(self):
        """Test L-BFGS optimizer initialization."""
        # Default initialization
        optimizer = LBFGSOptimizer()
        self.assertEqual(optimizer.bounds, [(0.1, 5.0)] * 3)
        self.assertEqual(optimizer.max_iterations, 50)

        # Custom initialization
        custom_optimizer = LBFGSOptimizer(
            bounds=[(0.5, 2.0)] * 3,
            max_iterations=100
        )
        self.assertEqual(custom_optimizer.bounds, [(0.5, 2.0)] * 3)
        self.assertEqual(custom_optimizer.max_iterations, 100)

    def test_optimization_convergence(self):
        """Test that L-BFGS finds the correct optimum."""
        optimal_weights = (1.2, 0.8, 1.5)
        eval_function = self.create_mock_evaluation_function(optimal_weights)

        # Run optimization
        result_weights = self.optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )

        # Should converge close to optimal weights
        alpha, beta, gamma = result_weights
        opt_alpha, opt_beta, opt_gamma = optimal_weights

        self.assertAlmostEqual(alpha, opt_alpha, delta=0.1)
        self.assertAlmostEqual(beta, opt_beta, delta=0.1)
        self.assertAlmostEqual(gamma, opt_gamma, delta=0.1)

    def test_bounds_enforcement(self):
        """Test that optimizer respects bounds."""
        # Set tight bounds
        bounded_optimizer = LBFGSOptimizer(bounds=[(0.5, 1.5)] * 3)

        eval_function = self.create_mock_evaluation_function((2.0, 0.3, 3.0))  # Outside bounds

        result_weights = bounded_optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )

        # All weights should be within bounds
        alpha, beta, gamma = result_weights
        self.assertGreaterEqual(alpha, 0.5)
        self.assertLessEqual(alpha, 1.5)
        self.assertGreaterEqual(beta, 0.5)
        self.assertLessEqual(beta, 1.5)
        self.assertGreaterEqual(gamma, 0.5)
        self.assertLessEqual(gamma, 1.5)

    def test_optimization_with_flat_function(self):
        """Test optimization with flat evaluation function."""

        def flat_eval_function(weights: Tuple[float, float, float]) -> float:
            return 0.5  # Constant function

        result_weights = self.optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=flat_eval_function
        )

        # Should return valid weights within bounds
        alpha, beta, gamma = result_weights
        self.assertGreaterEqual(alpha, 0.1)
        self.assertLessEqual(alpha, 5.0)
        self.assertGreaterEqual(beta, 0.1)
        self.assertLessEqual(beta, 5.0)
        self.assertGreaterEqual(gamma, 0.1)
        self.assertLessEqual(gamma, 5.0)

    def test_optimization_with_noise(self):
        """Test optimization robustness with noisy evaluation."""
        eval_function = self.create_noisy_evaluation_function(noise_level=0.02)

        # Run multiple times to test consistency
        results = []
        for _ in range(3):
            np.random.seed(42)  # For reproducibility
            result_weights = self.optimizer.optimize_weights(
                training_data=self.mock_training_data,
                validation_queries=self.mock_validation_queries,
                validation_qrels=self.mock_validation_qrels,
                evaluation_function=eval_function
            )
            results.append(result_weights)

        # Results should be reasonably stable
        alphas = [r[0] for r in results]
        betas = [r[1] for r in results]
        gammas = [r[2] for r in results]

        # Standard deviation should be reasonable (not too high)
        self.assertLess(np.std(alphas), 0.5)
        self.assertLess(np.std(betas), 0.5)
        self.assertLess(np.std(gammas), 0.5)


class TestGridSearchOptimizer(TestWeightOptimizerBase):
    """Test cases for Grid Search optimizer."""

    def setUp(self):
        """Set up Grid Search optimizer tests."""
        super().setUp()
        self.optimizer = GridSearchOptimizer(
            weight_ranges=[[0.5, 2.0]] * 3,
            resolution=5  # Small resolution for fast testing
        )

    def test_initialization(self):
        """Test Grid Search optimizer initialization."""
        # Default initialization
        optimizer = GridSearchOptimizer()
        self.assertEqual(len(optimizer.weight_ranges), 3)
        self.assertEqual(optimizer.resolution, 10)

        # Custom initialization
        custom_optimizer = GridSearchOptimizer(
            weight_ranges=[[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]],
            resolution=5
        )
        self.assertEqual(custom_optimizer.weight_ranges, [[0.1, 1.0], [0.2, 2.0], [0.3, 3.0]])
        self.assertEqual(custom_optimizer.resolution, 5)

    def test_grid_generation(self):
        """Test that grid points are generated correctly."""
        # Use small resolution to test exactly
        small_optimizer = GridSearchOptimizer(
            weight_ranges=[[0.0, 1.0]] * 3,
            resolution=3  # Should give points: 0.0, 0.5, 1.0
        )

        eval_function = self.create_mock_evaluation_function()

        # Run optimization
        result_weights = small_optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )

        # Result should be one of the grid points
        alpha, beta, gamma = result_weights
        expected_values = [0.0, 0.5, 1.0]

        self.assertIn(alpha, expected_values)
        self.assertIn(beta, expected_values)
        self.assertIn(gamma, expected_values)

    def test_exhaustive_search(self):
        """Test that grid search finds global optimum in discrete space."""

        # Create evaluation function with known optimum at grid point
        def discrete_eval_function(weights: Tuple[float, float, float]) -> float:
            alpha, beta, gamma = weights
            # Optimum at (1.0, 1.0, 1.0)
            if abs(alpha - 1.0) < 0.1 and abs(beta - 1.0) < 0.1 and abs(gamma - 1.0) < 0.1:
                return 1.0
            else:
                return 0.5

        optimizer = GridSearchOptimizer(
            weight_ranges=[[0.5, 1.5]] * 3,
            resolution=3  # Points: 0.5, 1.0, 1.5
        )

        result_weights = optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=discrete_eval_function
        )

        # Should find the optimum
        alpha, beta, gamma = result_weights
        self.assertAlmostEqual(alpha, 1.0, delta=0.1)
        self.assertAlmostEqual(beta, 1.0, delta=0.1)
        self.assertAlmostEqual(gamma, 1.0, delta=0.1)

    def test_performance_tracking(self):
        """Test that grid search tracks performance correctly."""
        eval_function = self.create_mock_evaluation_function()

        result_weights = self.optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )

        # Should have tracked evaluations
        self.assertGreater(len(self.optimizer.evaluation_history), 0)

        # Best score should correspond to result weights
        best_score = max(self.optimizer.evaluation_history)
        final_score = eval_function(result_weights)
        self.assertAlmostEqual(best_score, final_score, delta=1e-6)


class TestRandomSearchOptimizer(TestWeightOptimizerBase):
    """Test cases for Random Search optimizer."""

    def setUp(self):
        """Set up Random Search optimizer tests."""
        super().setUp()
        self.optimizer = RandomSearchOptimizer(
            num_samples=50,  # Small number for fast testing
            weight_ranges=[[0.1, 2.0]] * 3
        )

    def test_initialization(self):
        """Test Random Search optimizer initialization."""
        # Default initialization
        optimizer = RandomSearchOptimizer()
        self.assertEqual(optimizer.num_samples, 100)
        self.assertEqual(len(optimizer.weight_ranges), 3)

        # Custom initialization
        custom_optimizer = RandomSearchOptimizer(
            num_samples=200,
            weight_ranges=[[0.5, 1.5]] * 3
        )
        self.assertEqual(custom_optimizer.num_samples, 200)
        self.assertEqual(custom_optimizer.weight_ranges, [[0.5, 1.5]] * 3)

    def test_random_sampling(self):
        """Test that random sampling covers the search space."""
        np.random.seed(42)  # For reproducibility

        eval_function = self.create_mock_evaluation_function()

        result_weights = self.optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )

        # Result should be within bounds
        alpha, beta, gamma = result_weights
        self.assertGreaterEqual(alpha, 0.1)
        self.assertLessEqual(alpha, 2.0)
        self.assertGreaterEqual(beta, 0.1)
        self.assertLessEqual(beta, 2.0)
        self.assertGreaterEqual(gamma, 0.1)
        self.assertLessEqual(gamma, 2.0)

        # Should have evaluated the correct number of samples
        self.assertEqual(len(self.optimizer.evaluation_history), 50)

    def test_convergence_with_enough_samples(self):
        """Test that random search converges with sufficient samples."""
        # Use many samples for better convergence
        large_optimizer = RandomSearchOptimizer(
            num_samples=500,
            weight_ranges=[[0.5, 2.0]] * 3
        )

        optimal_weights = (1.2, 0.8, 1.5)
        eval_function = self.create_mock_evaluation_function(optimal_weights)

        np.random.seed(42)
        result_weights = large_optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )

        # Should be reasonably close to optimum (less precise than L-BFGS)
        alpha, beta, gamma = result_weights
        opt_alpha, opt_beta, opt_gamma = optimal_weights

        self.assertAlmostEqual(alpha, opt_alpha, delta=0.3)
        self.assertAlmostEqual(beta, opt_beta, delta=0.3)
        self.assertAlmostEqual(gamma, opt_gamma, delta=0.3)

    def test_reproducibility(self):
        """Test that random search is reproducible with fixed seed."""
        eval_function = self.create_mock_evaluation_function()

        # Run twice with same seed
        np.random.seed(42)
        result1 = self.optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )

        # Reset optimizer and run again
        self.optimizer = RandomSearchOptimizer(num_samples=50, weight_ranges=[[0.1, 2.0]] * 3)
        np.random.seed(42)
        result2 = self.optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )

        # Results should be identical
        self.assertEqual(result1, result2)


class TestOptimizerFactory(TestWeightOptimizerBase):
    """Test cases for optimizer factory function."""

    def test_create_optimizer_types(self):
        """Test that factory creates correct optimizer types."""
        # Test L-BFGS creation
        lbfgs_optimizer = create_optimizer('lbfgs')
        self.assertIsInstance(lbfgs_optimizer, LBFGSOptimizer)

        # Test Grid Search creation
        grid_optimizer = create_optimizer('grid')
        self.assertIsInstance(grid_optimizer, GridSearchOptimizer)

        # Test Random Search creation
        random_optimizer = create_optimizer('random')
        self.assertIsInstance(random_optimizer, RandomSearchOptimizer)

    def test_create_optimizer_invalid_type(self):
        """Test that factory handles invalid optimizer types."""
        with self.assertRaises(ValueError):
            create_optimizer('invalid_optimizer')

    def test_create_optimizer_with_kwargs(self):
        """Test that factory passes kwargs correctly."""
        # Test with custom parameters
        lbfgs_optimizer = create_optimizer('lbfgs', bounds=[(0.5, 2.0)] * 3, max_iterations=100)
        self.assertEqual(lbfgs_optimizer.bounds, [(0.5, 2.0)] * 3)
        self.assertEqual(lbfgs_optimizer.max_iterations, 100)

        grid_optimizer = create_optimizer('grid', resolution=5)
        self.assertEqual(grid_optimizer.resolution, 5)

        random_optimizer = create_optimizer('random', num_samples=200)
        self.assertEqual(random_optimizer.num_samples, 200)


class TestOptimizerComparison(TestWeightOptimizerBase):
    """Integration tests comparing different optimizers."""

    def test_optimizer_comparison_on_same_problem(self):
        """Test that all optimizers can solve the same problem."""
        optimal_weights = (1.0, 1.0, 1.0)  # Simple optimum
        eval_function = self.create_mock_evaluation_function(optimal_weights)

        optimizers = {
            'lbfgs': LBFGSOptimizer(bounds=[(0.5, 1.5)] * 3),
            'grid': GridSearchOptimizer(weight_ranges=[[0.5, 1.5]] * 3, resolution=5),
            'random': RandomSearchOptimizer(num_samples=100, weight_ranges=[[0.5, 1.5]] * 3)
        }

        results = {}
        for name, optimizer in optimizers.items():
            if name == 'random':
                np.random.seed(42)  # For reproducibility

            result_weights = optimizer.optimize_weights(
                training_data=self.mock_training_data,
                validation_queries=self.mock_validation_queries,
                validation_qrels=self.mock_validation_qrels,
                evaluation_function=eval_function
            )
            results[name] = result_weights

        # All should find reasonable solutions
        for name, weights in results.items():
            alpha, beta, gamma = weights
            with self.subTest(optimizer=name):
                self.assertAlmostEqual(alpha, 1.0, delta=0.3)
                self.assertAlmostEqual(beta, 1.0, delta=0.3)
                self.assertAlmostEqual(gamma, 1.0, delta=0.3)

    def test_optimization_performance_comparison(self):
        """Test performance characteristics of different optimizers."""
        eval_function = self.create_mock_evaluation_function()

        # Count function evaluations for each optimizer
        evaluation_counts = {}

        # L-BFGS (should be most efficient)
        lbfgs_optimizer = LBFGSOptimizer()
        eval_count = 0

        def counting_eval_lbfgs(weights):
            nonlocal eval_count
            eval_count += 1
            return eval_function(weights)

        lbfgs_optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=counting_eval_lbfgs
        )
        evaluation_counts['lbfgs'] = eval_count

        # Grid search (fixed number of evaluations)
        grid_optimizer = GridSearchOptimizer(resolution=3)  # 3^3 = 27 evaluations
        grid_optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )
        evaluation_counts['grid'] = len(grid_optimizer.evaluation_history)

        # Random search (fixed number of evaluations)
        random_optimizer = RandomSearchOptimizer(num_samples=30)
        random_optimizer.optimize_weights(
            training_data=self.mock_training_data,
            validation_queries=self.mock_validation_queries,
            validation_qrels=self.mock_validation_qrels,
            evaluation_function=eval_function
        )
        evaluation_counts['random'] = len(random_optimizer.evaluation_history)

        # Verify expected evaluation counts
        self.assertEqual(evaluation_counts['grid'], 27)  # 3^3
        self.assertEqual(evaluation_counts['random'], 30)  # num_samples
        self.assertLess(evaluation_counts['lbfgs'], 100)  # Should be efficient


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestWeightOptimizerBase))
    suite.addTests(loader.loadTestsFromTestCase(TestLBFGSOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestGridSearchOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomSearchOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizerFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizerComparison))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when script is executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Test weight optimizer functionality")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--test', '-t', type=str, default=None,
                        help='Run specific test method')
    parser.add_argument('--optimizer', '-o', type=str, default=None,
                        choices=['lbfgs', 'grid', 'random'],
                        help='Test specific optimizer only')

    args = parser.parse_args()

    if args.test:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTest(TestLBFGSOptimizer(args.test))
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
    elif args.optimizer:
        # Run tests for specific optimizer
        if args.optimizer == 'lbfgs':
            suite = unittest.TestSuite()
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLBFGSOptimizer))
        elif args.optimizer == 'grid':
            suite = unittest.TestSuite()
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGridSearchOptimizer))
        elif args.optimizer == 'random':
            suite = unittest.TestSuite()
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRandomSearchOptimizer))

        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
    else:
        # Run all tests
        success = run_all_tests()
        if not success:
            sys.exit(1)

    print(f"\nWeight optimizer tests completed!")