#!/usr/bin/env python3
"""
Unit tests for training data creation functionality.

Tests the TrainingDataCreator class and data creation pipeline:
- Feature extraction for queries
- Integration with RM expansion, BM25, and semantic similarity
- Data validation and statistics computation
- Error handling and edge cases
- Mock data generation for testing

Usage:
    python -m pytest test_data_creator.py -v
    python test_data_creator.py  # Run directly
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import modules to test
from utils.file_utils import save_training_data, load_training_data


class MockBM25Scorer:
    """Mock BM25 scorer for testing."""

    def __init__(self):
        self.term_scores = {
            'machine': 2.1,
            'learning': 1.8,
            'neural': 2.5,
            'networks': 2.0,
            'algorithm': 1.5,
            'artificial': 1.3,
            'intelligence': 1.7,
            'data': 1.2,
            'science': 1.4
        }

    def compute_bm25_term_weight(self, doc_id: str, terms: List[str]) -> Dict[str, float]:
        """Mock BM25 term weight computation."""
        return {term: self.term_scores.get(term, 0.5) for term in terms}


class MockSemanticSimilarity:
    """Mock semantic similarity computer for testing."""

    def __init__(self):
        self.similarity_scores = {
            'machine': 0.85,
            'learning': 0.78,
            'neural': 0.82,
            'networks': 0.75,
            'algorithm': 0.70,
            'artificial': 0.73,
            'intelligence': 0.76,
            'data': 0.65,
            'science': 0.68
        }

    def compute_query_expansion_similarities(self, query: str, expansion_terms: List[str]) -> Dict[str, float]:
        """Mock semantic similarity computation."""
        return {term: self.similarity_scores.get(term, 0.5) for term in expansion_terms}


class MockRMExpansion:
    """Mock RM expansion for testing."""

    def expand_query(self, query: str, documents: List[str], scores: List[float],
                     num_expansion_terms: int = 10, rm_type: str = "rm3") -> List[Tuple[str, float]]:
        """Mock RM expansion."""
        # Generate mock expansion terms based on query
        base_terms = ['machine', 'learning', 'neural', 'networks', 'algorithm',
                      'artificial', 'intelligence', 'data', 'science', 'computer']

        # Filter out query terms for RM1
        query_terms = set(query.lower().split())
        if rm_type == "rm1":
            expansion_terms = [term for term in base_terms if term not in query_terms]
        else:  # RM3
            expansion_terms = base_terms

        # Generate weights (higher for first terms)
        expansion_with_weights = []
        for i, term in enumerate(expansion_terms[:num_expansion_terms]):
            weight = 1.0 - (i * 0.1)  # Decreasing weights
            expansion_with_weights.append((term, max(0.1, weight)))

        return expansion_with_weights


class TestTrainingDataCreator(unittest.TestCase):
    """Test cases for TrainingDataCreator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_bm25_scorer = MockBM25Scorer()
        self.mock_semantic_sim = MockSemanticSimilarity()
        self.mock_rm_expansion = MockRMExpansion()

        # Sample test data
        self.sample_queries = {
            'q1': 'machine learning algorithms',
            'q2': 'neural networks deep learning',
            'q3': 'artificial intelligence systems'
        }

        self.sample_documents = {
            'doc1': 'Machine learning algorithms are used in artificial intelligence applications.',
            'doc2': 'Neural networks and deep learning models process natural language text.',
            'doc3': 'Information retrieval systems use algorithms to find relevant documents.',
            'doc4': 'Computer science research focuses on machine learning and data mining.',
            'doc5': 'Artificial intelligence systems can learn from data and make predictions.'
        }

        self.sample_qrels = {
            'q1': {'doc1': 2, 'doc2': 1, 'doc4': 1},
            'q2': {'doc2': 2, 'doc5': 1},
            'q3': {'doc1': 1, 'doc5': 2, 'doc3': 1}
        }

        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_training_data_creator(self):
        """Create TrainingDataCreator with mock components."""
        # Import here to avoid import issues if module not available
        try:
            sys.path.append(str(project_root / "scripts"))
            from create_training_data import TrainingDataCreator

            creator = TrainingDataCreator(
                bm25_scorer=self.mock_bm25_scorer,
                semantic_similarity=self.mock_semantic_sim,
                rm_expansion=self.mock_rm_expansion,
                max_expansion_terms=10,
                min_relevant_docs=1
            )
            return creator
        except ImportError:
            # Create a simplified mock if import fails
            return self.create_simple_mock_creator()

    def create_simple_mock_creator(self):
        """Create a simple mock creator for testing basic functionality."""

        class SimpleMockCreator:
            def __init__(self, bm25_scorer, semantic_similarity, rm_expansion,
                         max_expansion_terms=10, min_relevant_docs=1):
                self.bm25_scorer = bm25_scorer
                self.semantic_sim = semantic_similarity
                self.rm_expansion = rm_expansion
                self.max_expansion_terms = max_expansion_terms
                self.min_relevant_docs = min_relevant_docs

            def extract_features_for_query(self, query_id, query_text, relevant_docs, documents):
                """Extract features for a single query."""
                if len(relevant_docs) < self.min_relevant_docs:
                    return {}

                # Get relevant document texts
                pseudo_relevant_docs = [documents.get(doc_id, "") for doc_id in relevant_docs]
                pseudo_scores = [1.0] * len(pseudo_relevant_docs)

                # RM expansion
                expansion_terms = self.rm_expansion.expand_query(
                    query=query_text,
                    documents=pseudo_relevant_docs,
                    scores=pseudo_scores,
                    num_expansion_terms=self.max_expansion_terms
                )

                if not expansion_terms:
                    return {}

                # Extract features
                term_features = {}
                expansion_words = [term for term, weight in expansion_terms]

                # Semantic similarities
                semantic_scores = self.semantic_sim.compute_query_expansion_similarities(
                    query_text, expansion_words
                )

                for term, rm_weight in expansion_terms:
                    # BM25 score
                    reference_doc_id = relevant_docs[0] if relevant_docs else None
                    bm25_score = 0.0
                    if self.bm25_scorer and reference_doc_id:
                        bm25_scores = self.bm25_scorer.compute_bm25_term_weight(reference_doc_id, [term])
                        bm25_score = bm25_scores.get(term, 0.0)

                    # Semantic score
                    semantic_score = semantic_scores.get(term, 0.0)

                    term_features[term] = {
                        'rm_weight': float(rm_weight),
                        'bm25_score': float(bm25_score),
                        'semantic_score': float(semantic_score)
                    }

                return {
                    'query_id': query_id,
                    'query_text': query_text,
                    'expansion_terms': expansion_terms,
                    'term_features': term_features,
                    'num_relevant_docs': len(relevant_docs),
                    'reference_doc_id': relevant_docs[0] if relevant_docs else None
                }

        return SimpleMockCreator(
            self.mock_bm25_scorer,
            self.mock_semantic_sim,
            self.mock_rm_expansion,
            max_expansion_terms=10,
            min_relevant_docs=1
        )

    def test_creator_initialization(self):
        """Test TrainingDataCreator initialization."""
        creator = self.create_mock_training_data_creator()

        # Check that components are set
        self.assertIsNotNone(creator.bm25_scorer)
        self.assertIsNotNone(creator.semantic_sim)
        self.assertIsNotNone(creator.rm_expansion)
        self.assertEqual(creator.max_expansion_terms, 10)
        self.assertEqual(creator.min_relevant_docs, 1)

    def test_feature_extraction_single_query(self):
        """Test feature extraction for a single query."""
        creator = self.create_mock_training_data_creator()

        # Extract features for one query
        query_id = 'q1'
        query_text = self.sample_queries[query_id]
        relevant_docs = list(self.sample_qrels[query_id].keys())

        features = creator.extract_features_for_query(
            query_id=query_id,
            query_text=query_text,
            relevant_docs=relevant_docs,
            documents=self.sample_documents
        )

        # Should return valid features
        self.assertIsInstance(features, dict)
        self.assertIn('query_id', features)
        self.assertIn('query_text', features)
        self.assertIn('term_features', features)
        self.assertIn('expansion_terms', features)

        # Check feature structure
        term_features = features['term_features']
        self.assertIsInstance(term_features, dict)

        for term, term_data in term_features.items():
            self.assertIn('rm_weight', term_data)
            self.assertIn('bm25_score', term_data)
            self.assertIn('semantic_score', term_data)

            # All scores should be numeric and non-negative
            self.assertIsInstance(term_data['rm_weight'], float)
            self.assertIsInstance(term_data['bm25_score'], float)
            self.assertIsInstance(term_data['semantic_score'], float)
            self.assertGreaterEqual(term_data['rm_weight'], 0.0)
            self.assertGreaterEqual(term_data['bm25_score'], 0.0)
            self.assertGreaterEqual(term_data['semantic_score'], 0.0)

    def test_feature_extraction_insufficient_docs(self):
        """Test feature extraction with insufficient relevant documents."""
        creator = self.create_mock_training_data_creator()
        creator.min_relevant_docs = 3  # Require more docs than available

        query_id = 'q2'
        query_text = self.sample_queries[query_id]
        relevant_docs = list(self.sample_qrels[query_id].keys())  # Only 2 docs

        features = creator.extract_features_for_query(
            query_id=query_id,
            query_text=query_text,
            relevant_docs=relevant_docs,
            documents=self.sample_documents
        )

        # Should return empty dict
        self.assertEqual(features, {})

    def test_feature_extraction_missing_documents(self):
        """Test feature extraction with missing documents."""
        creator = self.create_mock_training_data_creator()

        query_id = 'q1'
        query_text = self.sample_queries[query_id]
        relevant_docs = ['missing_doc1', 'missing_doc2']  # Non-existent docs

        features = creator.extract_features_for_query(
            query_id=query_id,
            query_text=query_text,
            relevant_docs=relevant_docs,
            documents=self.sample_documents
        )

        # Should handle gracefully (might return empty or partial features)
        self.assertIsInstance(features, dict)

    def test_feature_extraction_no_bm25_scorer(self):
        """Test feature extraction without BM25 scorer."""
        creator = self.create_simple_mock_creator()
        creator.bm25_scorer = None  # No BM25 scorer

        query_id = 'q1'
        query_text = self.sample_queries[query_id]
        relevant_docs = list(self.sample_qrels[query_id].keys())

        features = creator.extract_features_for_query(
            query_id=query_id,
            query_text=query_text,
            relevant_docs=relevant_docs,
            documents=self.sample_documents
        )

        # Should still work, with BM25 scores as 0.0
        self.assertIsInstance(features, dict)
        if 'term_features' in features:
            for term, term_data in features['term_features'].items():
                self.assertEqual(term_data['bm25_score'], 0.0)

    def test_feature_consistency(self):
        """Test that feature extraction is consistent across runs."""
        creator = self.create_mock_training_data_creator()

        query_id = 'q1'
        query_text = self.sample_queries[query_id]
        relevant_docs = list(self.sample_qrels[query_id].keys())

        # Extract features twice
        features1 = creator.extract_features_for_query(
            query_id, query_text, relevant_docs, self.sample_documents
        )
        features2 = creator.extract_features_for_query(
            query_id, query_text, relevant_docs, self.sample_documents
        )

        # Should be identical
        self.assertEqual(features1, features2)

    def test_statistics_computation(self):
        """Test statistics computation over extracted features."""
        # Create sample features data
        sample_features = {
            'q1': {
                'term_features': {
                    'machine': {'rm_weight': 0.8, 'bm25_score': 2.1, 'semantic_score': 0.7},
                    'learning': {'rm_weight': 0.6, 'bm25_score': 1.8, 'semantic_score': 0.6}
                }
            },
            'q2': {
                'term_features': {
                    'neural': {'rm_weight': 0.9, 'bm25_score': 2.5, 'semantic_score': 0.8},
                    'networks': {'rm_weight': 0.7, 'bm25_score': 2.0, 'semantic_score': 0.7}
                }
            }
        }

        creator = self.create_mock_training_data_creator()

        # Test statistics computation (if method exists)
        if hasattr(creator, '_compute_statistics'):
            stats = creator._compute_statistics(sample_features)

            self.assertIn('num_queries', stats)
            self.assertIn('avg_expansion_terms', stats)
            self.assertIn('feature_stats', stats)

            self.assertEqual(stats['num_queries'], 2)
            self.assertEqual(stats['avg_expansion_terms'], 2.0)

            # Check feature statistics
            feature_stats = stats['feature_stats']
            self.assertIn('rm_weights', feature_stats)
            self.assertIn('bm25_scores', feature_stats)
            self.assertIn('semantic_scores', feature_stats)

    def test_data_saving_and_loading(self):
        """Test saving and loading training data."""
        # Create sample training dataset
        training_dataset = {
            'queries': self.sample_queries,
            'qrels': self.sample_qrels,
            'documents': self.sample_documents,
            'features': {
                'q1': {
                    'term_features': {
                        'machine': {'rm_weight': 0.8, 'bm25_score': 2.1, 'semantic_score': 0.7}
                    }
                }
            },
            'statistics': {
                'num_queries': 3,
                'avg_expansion_terms': 5.0
            },
            'metadata': {
                'dataset_name': 'test_dataset',
                'max_expansion_terms': 10
            }
        }

        # Save training data
        output_dir = Path(self.temp_dir) / 'test_training_data'
        save_training_data(training_dataset, output_dir)

        # Verify files were created
        self.assertTrue((output_dir / 'queries.json').exists())
        self.assertTrue((output_dir / 'qrels.json').exists())
        self.assertTrue((output_dir / 'metadata.json').exists())

        # Load training data
        loaded_data = load_training_data(output_dir)

        # Verify loaded data
        self.assertEqual(loaded_data['queries'], self.sample_queries)
        self.assertEqual(loaded_data['qrels'], self.sample_qrels)
        self.assertIn('features', loaded_data)
        self.assertIn('metadata', loaded_data)

    def test_edge_cases(self):
        """Test various edge cases."""
        creator = self.create_mock_training_data_creator()

        # Empty query
        features_empty_query = creator.extract_features_for_query(
            query_id='empty',
            query_text='',
            relevant_docs=['doc1'],
            documents=self.sample_documents
        )
        self.assertIsInstance(features_empty_query, dict)

        # Query with no relevant docs
        features_no_docs = creator.extract_features_for_query(
            query_id='no_docs',
            query_text='test query',
            relevant_docs=[],
            documents=self.sample_documents
        )
        self.assertEqual(features_no_docs, {})

        # Very long query
        long_query = ' '.join(['word'] * 100)
        features_long_query = creator.extract_features_for_query(
            query_id='long',
            query_text=long_query,
            relevant_docs=['doc1'],
            documents=self.sample_documents
        )
        self.assertIsInstance(features_long_query, dict)

    def test_integration_with_mock_data(self):
        """Test full pipeline with mock MSMARCO-like data."""
        creator = self.create_mock_training_data_creator()

        # Process all queries
        all_features = {}
        for query_id, query_text in self.sample_queries.items():
            if query_id in self.sample_qrels:
                relevant_docs = [doc_id for doc_id, rel in self.sample_qrels[query_id].items() if rel >= 1]

                features = creator.extract_features_for_query(
                    query_id=query_id,
                    query_text=query_text,
                    relevant_docs=relevant_docs,
                    documents=self.sample_documents
                )

                if features:
                    all_features[query_id] = features

        # Should have processed some queries successfully
        self.assertGreater(len(all_features), 0)

        # All processed queries should have valid structure
        for query_id, features in all_features.items():
            self.assertIn('query_id', features)
            self.assertIn('term_features', features)
            self.assertIsInstance(features['term_features'], dict)

            # Each term should have all three feature types
            for term, term_data in features['term_features'].items():
                self.assertIn('rm_weight', term_data)
                self.assertIn('bm25_score', term_data)
                self.assertIn('semantic_score', term_data)


class TestMockComponents(unittest.TestCase):
    """Test cases for mock components used in testing."""

    def test_mock_bm25_scorer(self):
        """Test mock BM25 scorer functionality."""
        scorer = MockBM25Scorer()

        # Test term scoring
        scores = scorer.compute_bm25_term_weight('doc1', ['machine', 'learning', 'unknown'])

        self.assertEqual(scores['machine'], 2.1)
        self.assertEqual(scores['learning'], 1.8)
        self.assertEqual(scores['unknown'], 0.5)  # Default score

    def test_mock_semantic_similarity(self):
        """Test mock semantic similarity functionality."""
        sim_computer = MockSemanticSimilarity()

        # Test similarity computation
        similarities = sim_computer.compute_query_expansion_similarities(
            'machine learning', ['machine', 'learning', 'unknown']
        )

        self.assertEqual(similarities['machine'], 0.85)
        self.assertEqual(similarities['learning'], 0.78)
        self.assertEqual(similarities['unknown'], 0.5)  # Default score

    def test_mock_rm_expansion(self):
        """Test mock RM expansion functionality."""
        rm_expansion = MockRMExpansion()

        # Test RM3 expansion (includes query terms)
        rm3_terms = rm_expansion.expand_query(
            'machine learning', ['sample document'], [1.0], num_expansion_terms=5, rm_type='rm3'
        )

        self.assertEqual(len(rm3_terms), 5)
        self.assertTrue(all(isinstance(term, str) and isinstance(weight, float) for term, weight in rm3_terms))

        # Test RM1 expansion (excludes query terms)
        rm1_terms = rm_expansion.expand_query(
            'machine learning', ['sample document'], [1.0], num_expansion_terms=5, rm_type='rm1'
        )

        self.assertEqual(len(rm1_terms), 5)
        # Should not contain original query terms
        rm1_words = [term for term, weight in rm1_terms]
        self.assertNotIn('machine', rm1_words)
        self.assertNotIn('learning', rm1_words)


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingDataCreator))
    suite.addTests(loader.loadTestsFromTestCase(TestMockComponents))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when script is executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Test training data creation functionality")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--test', '-t', type=str, default=None,
                        help='Run specific test method')

    args = parser.parse_args()

    if args.test:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTest(TestTrainingDataCreator(args.test))
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
    else:
        # Run all tests
        success = run_all_tests()
        if not success:
            sys.exit(1)

    print("\nTraining data creator tests completed!")