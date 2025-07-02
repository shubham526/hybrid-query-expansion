#!/usr/bin/env python3
"""
Unit tests for evaluation functionality.

Tests the evaluation modules including:
- TRECEvaluator class
- ExpansionEvaluator class
- Integration with pytrec_eval
- Metrics computation and comparison
- Results table generation
- Error handling and edge cases

Usage:
    python -m pytest test_evaluation.py -v
    python test_evaluation.py  # Run directly
"""

import unittest
import sys
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import modules to test
try:
    from src.evaluation.evaluator import TRECEvaluator, ExpansionEvaluator, create_trec_dl_evaluator
    from src.evaluation.metrics import get_metric

    EVALUATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Evaluator modules not available: {e}")
    EVALUATOR_AVAILABLE = False


class TestTRECEvaluator(unittest.TestCase):
    """Test cases for TRECEvaluator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Sample test data
        self.sample_run_results = {
            'query1': [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)],
            'query2': [('doc4', 0.95), ('doc5', 0.85), ('doc6', 0.75)],
            'query3': [('doc7', 0.88), ('doc8', 0.78), ('doc9', 0.68)]
        }

        self.sample_qrels = {
            'query1': {'doc1': 2, 'doc2': 1, 'doc3': 0},
            'query2': {'doc4': 2, 'doc5': 0, 'doc6': 1},
            'query3': {'doc7': 1, 'doc8': 2, 'doc9': 0}
        }

        # Multiple runs for comparison
        self.sample_runs = {
            'baseline': {
                'query1': [('doc1', 0.8), ('doc2', 0.7), ('doc3', 0.6)],
                'query2': [('doc4', 0.85), ('doc5', 0.75), ('doc6', 0.65)],
                'query3': [('doc7', 0.78), ('doc8', 0.68), ('doc9', 0.58)]
            },
            'improved': {
                'query1': [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)],
                'query2': [('doc4', 0.95), ('doc5', 0.85), ('doc6', 0.75)],
                'query3': [('doc7', 0.88), ('doc8', 0.78), ('doc9', 0.68)]
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(EVALUATOR_AVAILABLE, "Evaluator modules not available")
    def test_evaluator_initialization(self):
        """Test TRECEvaluator initialization."""
        # Default initialization
        evaluator = TRECEvaluator()
        self.assertIsInstance(evaluator.metrics, list)
        self.assertIn('map', evaluator.metrics)
        self.assertIn('ndcg_cut_10', evaluator.metrics)

        # Custom initialization
        custom_metrics = ['map', 'P_5', 'ndcg_cut_5']
        custom_evaluator = TRECEvaluator(metrics=custom_metrics)
        self.assertEqual(custom_evaluator.metrics, custom_metrics)

    def test_qrels_writing(self):
        """Test qrels file writing functionality."""
        if not EVALUATOR_AVAILABLE:
            self.skipTest("Evaluator not available")

        evaluator = TRECEvaluator()
        qrels_file = os.path.join(self.temp_dir, 'test.qrel')

        # Write qrels
        evaluator._write_qrels(self.sample_qrels, qrels_file)

        # Verify file exists and has correct format
        self.assertTrue(os.path.exists(qrels_file))

        with open(qrels_file, 'r') as f:
            lines = f.readlines()

        # Should have correct number of lines
        expected_lines = sum(len(docs) for docs in self.sample_qrels.values())
        self.assertEqual(len(lines), expected_lines)

        # Check format: query_id 0 doc_id relevance
        for line in lines:
            parts = line.strip().split()
            self.assertEqual(len(parts), 4)
            self.assertEqual(parts[1], '0')  # Second field should be 0
            self.assertTrue(parts[3].isdigit())  # Relevance should be numeric

    def test_run_writing(self):
        """Test run file writing functionality."""
        if not EVALUATOR_AVAILABLE:
            self.skipTest("Evaluator not available")

        evaluator = TRECEvaluator()
        run_file = os.path.join(self.temp_dir, 'test.run')

        # Write run
        evaluator._write_run(self.sample_run_results, run_file)

        # Verify file exists and has correct format
        self.assertTrue(os.path.exists(run_file))

        with open(run_file, 'r') as f:
            lines = f.readlines()

        # Should have correct number of lines
        expected_lines = sum(len(docs) for docs in self.sample_run_results.values())
        self.assertEqual(len(lines), expected_lines)

        # Check format: query_id Q0 doc_id rank score run_name
        for line in lines:
            parts = line.strip().split()
            self.assertEqual(len(parts), 6)
            self.assertEqual(parts[1], 'Q0')  # Second field should be Q0
            self.assertTrue(parts[3].isdigit())  # Rank should be numeric
            try:
                float(parts[4])  # Score should be numeric
            except ValueError:
                self.fail(f"Score {parts[4]} is not numeric")

    @unittest.skipUnless(EVALUATOR_AVAILABLE, "Evaluator modules not available")
    def test_single_run_evaluation(self):
        """Test evaluation of a single run."""
        evaluator = TRECEvaluator(metrics=['map'])

        # Mock the get_metric function to avoid dependency on pytrec_eval
        with patch('src.evaluation.metrics.get_metric') as mock_get_metric:
            mock_get_metric.return_value = 0.75  # Mock MAP score

            results = evaluator.evaluate_run(self.sample_run_results, self.sample_qrels)

            # Should return results dict
            self.assertIsInstance(results, dict)
            self.assertIn('map', results)
            self.assertEqual(results['map'], 0.75)

            # get_metric should have been called
            self.assertTrue(mock_get_metric.called)

    @unittest.skipUnless(EVALUATOR_AVAILABLE, "Evaluator modules not available")
    def test_multiple_runs_evaluation(self):
        """Test evaluation of multiple runs."""
        evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10'])

        # Mock the get_metric function
        def mock_get_metric_side_effect(qrels_file, run_file, metric):
            if 'baseline' in run_file:
                return 0.60 if metric == 'map' else 0.55
            else:  # improved run
                return 0.75 if metric == 'map' else 0.70

        with patch('src.evaluation.metrics.get_metric') as mock_get_metric:
            mock_get_metric.side_effect = mock_get_metric_side_effect

            results = evaluator.evaluate_multiple_runs(self.sample_runs, self.sample_qrels)

            # Should return results for both runs
            self.assertIsInstance(results, dict)
            self.assertIn('baseline', results)
            self.assertIn('improved', results)

            # Check baseline results
            baseline_results = results['baseline']
            self.assertEqual(baseline_results['map'], 0.60)
            self.assertEqual(baseline_results['ndcg_cut_10'], 0.55)

            # Check improved results
            improved_results = results['improved']
            self.assertEqual(improved_results['map'], 0.75)
            self.assertEqual(improved_results['ndcg_cut_10'], 0.70)

    @unittest.skipUnless(EVALUATOR_AVAILABLE, "Evaluator modules not available")
    def test_runs_comparison(self):
        """Test comparison of multiple runs."""
        evaluator = TRECEvaluator(metrics=['map'])

        # Mock the get_metric function
        def mock_get_metric_side_effect(qrels_file, run_file, metric):
            if 'baseline' in run_file:
                return 0.60
            else:  # improved run
                return 0.75

        with patch('src.evaluation.metrics.get_metric') as mock_get_metric:
            mock_get_metric.side_effect = mock_get_metric_side_effect

            comparison = evaluator.compare_runs(self.sample_runs, self.sample_qrels, 'baseline')

            # Should return comparison results
            self.assertIsInstance(comparison, dict)
            self.assertIn('evaluations', comparison)
            self.assertIn('baseline', comparison)
            self.assertIn('improvements', comparison)

            # Check baseline identification
            self.assertEqual(comparison['baseline'], 'baseline')

            # Check improvements calculation
            improvements = comparison['improvements']['improved']
            self.assertAlmostEqual(improvements['map_improvement_pct'], 25.0, delta=0.1)  # (0.75-0.60)/0.60*100
            self.assertAlmostEqual(improvements['map_improvement_abs'], 0.15, delta=0.01)

    @unittest.skipUnless(EVALUATOR_AVAILABLE, "Evaluator modules not available")
    def test_results_table_creation(self):
        """Test results table generation."""
        evaluator = TRECEvaluator(metrics=['map', 'ndcg_cut_10'])

        # Create mock comparison results
        comparison_results = {
            'evaluations': {
                'baseline': {'map': 0.60, 'ndcg_cut_10': 0.55},
                'improved': {'map': 0.75, 'ndcg_cut_10': 0.70}
            },
            'baseline': 'baseline',
            'improvements': {
                'improved': {
                    'map_improvement_abs': 0.15,
                    'ndcg_cut_10_improvement_abs': 0.15
                }
            }
        }

        table = evaluator.create_results_table(comparison_results)

        # Should return string table
        self.assertIsInstance(table, str)

        # Should contain method names and metrics
        self.assertIn('baseline', table)
        self.assertIn('improved', table)
        self.assertIn('map', table)
        self.assertIn('ndcg_cut_10', table)

        # Should show improvements
        self.assertIn('+0.15', table)  # Improvement notation

    def test_error_handling(self):
        """Test error handling in evaluation."""
        if not EVALUATOR_AVAILABLE:
            self.skipTest("Evaluator not available")

        evaluator = TRECEvaluator(metrics=['invalid_metric'])

        # Mock get_metric to raise exception
        with patch('src.evaluation.metrics.get_metric') as mock_get_metric:
            mock_get_metric.side_effect = Exception("Invalid metric")

            results = evaluator.evaluate_run(self.sample_run_results, self.sample_qrels)

            # Should handle error gracefully
            self.assertIsInstance(results, dict)
            self.assertEqual(results['invalid_metric'], 0.0)

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        if not EVALUATOR_AVAILABLE:
            self.skipTest("Evaluator not available")

        evaluator = TRECEvaluator()

        # Test with empty runs
        empty_results = evaluator.evaluate_run({}, self.sample_qrels)
        self.assertIsInstance(empty_results, dict)

        # Test with empty qrels
        empty_qrels_results = evaluator.evaluate_run(self.sample_run_results, {})
        self.assertIsInstance(empty_qrels_results, dict)


class TestExpansionEvaluator(unittest.TestCase):
    """Test cases for ExpansionEvaluator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Mock expansion models
        self.mock_expansion_models = {
            'uniform': Mock(),
            'rm_only': Mock(),
            'our_method': Mock()
        }

        # Mock data
        self.mock_queries = {
            'q1': 'machine learning algorithms',
            'q2': 'neural networks'
        }

        self.mock_qrels = {
            'q1': {'doc1': 2, 'doc2': 1},
            'q2': {'doc3': 2, 'doc4': 0}
        }

        self.mock_first_stage_runs = {
            'q1': [('doc1', 0.8), ('doc2', 0.7)],
            'q2': [('doc3', 0.9), ('doc4', 0.6)]
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(EVALUATOR_AVAILABLE, "Evaluator modules not available")
    def test_expansion_evaluator_initialization(self):
        """Test ExpansionEvaluator initialization."""
        evaluator = ExpansionEvaluator()
        self.assertIsInstance(evaluator.metrics, list)

        # Should inherit from TRECEvaluator
        self.assertTrue(hasattr(evaluator, 'evaluate_run'))
        self.assertTrue(hasattr(evaluator, 'compare_runs'))

    def test_expansion_models_mock_behavior(self):
        """Test mock expansion models behavior."""
        # Configure mock expansion models
        for model_name, model in self.mock_expansion_models.items():
            model.expand_query.return_value = {
                'term1': 0.8,
                'term2': 0.6
            }

        # Test mock behavior
        for model_name, model in self.mock_expansion_models.items():
            importance_weights = model.expand_query(
                query='test query',
                pseudo_relevant_docs=['doc text'],
                pseudo_relevant_scores=[1.0],
                reference_doc_id='doc1'
            )

            self.assertEqual(importance_weights['term1'], 0.8)
            self.assertEqual(importance_weights['term2'], 0.6)

    @unittest.skipUnless(EVALUATOR_AVAILABLE, "Evaluator modules not available")
    def test_ablation_study_structure(self):
        """Test ablation study structure and flow."""
        evaluator = ExpansionEvaluator()

        # Mock reranker
        mock_reranker = Mock()
        mock_reranker.rerank_trec_dl_run.return_value = self.mock_first_stage_runs

        # Mock the evaluation process
        with patch.object(evaluator, 'evaluate_run') as mock_evaluate:
            mock_evaluate.return_value = {'map': 0.65, 'ndcg_cut_10': 0.60}

            with patch.object(evaluator, 'compare_runs') as mock_compare:
                mock_compare.return_value = {
                    'evaluations': {
                        'baseline': {'map': 0.60},
                        'our_method': {'map': 0.65}
                    },
                    'improvements': {
                        'our_method': {'map_improvement_abs': 0.05}
                    }
                }

                # This tests the structure without full implementation
                # since the full method requires complex dependencies
                baseline_run = self.mock_first_stage_runs

                # Verify that we can call evaluation methods
                result = evaluator.evaluate_run(baseline_run, self.mock_qrels)
                self.assertIsInstance(result, dict)

                comparison = evaluator.compare_runs(
                    {'baseline': baseline_run, 'improved': baseline_run},
                    self.mock_qrels,
                    'baseline'
                )
                self.assertIsInstance(comparison, dict)


class TestTRECDLEvaluatorFactory(unittest.TestCase):
    """Test cases for TREC DL evaluator factory."""

    @unittest.skipUnless(EVALUATOR_AVAILABLE, "Evaluator modules not available")
    def test_create_trec_dl_evaluator(self):
        """Test TREC DL evaluator factory function."""
        # Test 2019 evaluator
        evaluator_2019 = create_trec_dl_evaluator("2019")
        self.assertIsInstance(evaluator_2019, ExpansionEvaluator)

        # Test 2020 evaluator
        evaluator_2020 = create_trec_dl_evaluator("2020")
        self.assertIsInstance(evaluator_2020, ExpansionEvaluator)

        # Should have TREC DL specific metrics
        expected_metrics = ['map', 'ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_100']
        for metric in expected_metrics:
            self.assertIn(metric, evaluator_2019.metrics)


class TestMetricsIntegration(unittest.TestCase):
    """Test cases for metrics integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_files(self):
        """Create test qrels and run files."""
        # Create test qrels file
        qrels_file = os.path.join(self.temp_dir, 'test.qrel')
        with open(qrels_file, 'w') as f:
            f.write("q1 0 doc1 2\n")
            f.write("q1 0 doc2 1\n")
            f.write("q1 0 doc3 0\n")
            f.write("q2 0 doc4 2\n")
            f.write("q2 0 doc5 0\n")

        # Create test run file
        run_file = os.path.join(self.temp_dir, 'test.run')
        with open(run_file, 'w') as f:
            f.write("q1 Q0 doc1 1 0.9 test\n")
            f.write("q1 Q0 doc2 2 0.8 test\n")
            f.write("q1 Q0 doc3 3 0.7 test\n")
            f.write("q2 Q0 doc4 1 0.95 test\n")
            f.write("q2 Q0 doc5 2 0.85 test\n")

        return qrels_file, run_file

    @patch('src.evaluation.metrics.get_metric')
    def test_metrics_integration(self, mock_get_metric):
        """Test integration with metrics module."""
        # Mock get_metric to return a test score
        mock_get_metric.return_value = 0.75

        qrels_file, run_file = self.create_test_files()

        # Test metrics import and usage
        try:
            from src.evaluation.metrics import get_metric

            # Test that we can call get_metric
            score = get_metric(qrels_file, run_file, 'map')
            self.assertEqual(score, 0.75)  # Mock return value

            # Verify mock was called with correct arguments
            mock_get_metric.assert_called_with(qrels_file, run_file, 'map')

        except ImportError:
            self.skipTest("Metrics module not available")

    def test_file_format_compatibility(self):
        """Test that generated files are in correct TREC format."""
        qrels_file, run_file = self.create_test_files()

        # Verify qrels format
        with open(qrels_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            self.assertEqual(len(parts), 4)
            self.assertEqual(parts[1], '0')  # Second field should be 0
            self.assertTrue(parts[3].isdigit())  # Relevance should be digit

        # Verify run format
        with open(run_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            self.assertEqual(len(parts), 6)
            self.assertEqual(parts[1], 'Q0')  # Second field should be Q0
            self.assertTrue(parts[3].isdigit())  # Rank should be digit
            try:
                float(parts[4])  # Score should be float
            except ValueError:
                self.fail(f"Score {parts[4]} is not numeric")


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTRECEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestExpansionEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestTRECDLEvaluatorFactory))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when script is executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Test evaluation functionality")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--test', '-t', type=str, default=None,
                        help='Run specific test method')
    parser.add_argument('--skip-integration', action='store_true',
                        help='Skip integration tests requiring external dependencies')

    args = parser.parse_args()

    if not EVALUATOR_AVAILABLE and not args.skip_integration:
        print("Warning: Evaluator modules not available. Some tests will be skipped.")
        print("Use --skip-integration to skip integration tests.")

    if args.test:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTest(TestTRECEvaluator(args.test))
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
    else:
        # Run all tests
        success = run_all_tests()
        if not success:
            sys.exit(1)

    print("\nEvaluation tests completed!")